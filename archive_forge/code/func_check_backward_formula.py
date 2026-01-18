import torch
from torch import Tensor
import itertools
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten
from torch.utils import _pytree as pytree
from functools import partial
from torch.utils._mode_utils import no_dispatch, all_same_mode
import torch.autograd.forward_ad as fwAD
from typing import Callable
import re
def check_backward_formula(op: Callable, args, kwargs, output_process_fn_grad=None, gradcheck_wrapper=None, assert_equal_fn=None):
    CCT, cct_mode = generate_cct_and_mode()
    expected = compute_expected_grads(op, args, kwargs, output_process_fn_grad, gradcheck_wrapper)
    for choice in generate_subclass_choices_args_kwargs(args, kwargs, CCT, cct_mode):
        new_args, new_kwargs, which_args_are_wrapped, which_kwargs_are_wrapped = choice
        leaf_tensors = gather_leaf_tensors(new_args, new_kwargs)
        assert len(leaf_tensors) > 0
        try:
            if gradcheck_wrapper is None:
                results = op(*new_args, **new_kwargs)
            else:
                results = gradcheck_wrapper(op, *new_args, **new_kwargs)
            if output_process_fn_grad is not None:
                results = output_process_fn_grad(results)
        except RuntimeError as err:
            raise_composite_compliance_error(err, f'- wrapped_args: {which_args_are_wrapped}\n- wrapped_kwargs: {which_kwargs_are_wrapped}\n')
        flat_results = pytree.tree_leaves(results)
        flat_results = [r for r in flat_results if isinstance(r, torch.Tensor)]
        flat_diff_results = [r for r in flat_results if r.requires_grad]
        assert len(flat_diff_results) > 0
        grads = [torch.ones(r.shape, device=r.device, dtype=r.dtype) for r in flat_diff_results]
        for flat_new_grads, which_grad_is_batched in generate_subclass_choices(grads, CCT, cct_mode):
            try:
                actual = torch.autograd.grad(flat_diff_results, leaf_tensors, flat_new_grads, allow_unused=True, retain_graph=True)
            except RuntimeError as err:
                raise_composite_compliance_error(err, f'- wrapped_args: {which_args_are_wrapped}\n- wrapped_kwargs: {which_kwargs_are_wrapped}\n- wrapped_grads: {which_grad_is_batched}\n')

            def unwrap(e):
                return e.elem if isinstance(e, CCT) else e
            assert_equal_fn(tuple(map(unwrap, actual)), expected, equal_nan=True)
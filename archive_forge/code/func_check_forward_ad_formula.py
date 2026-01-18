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
def check_forward_ad_formula(op: Callable, args, kwargs, gradcheck_wrapper=None, assert_equal_fn=None):
    CCT, cct_mode = generate_cct_and_mode(autograd_view_consistency=False)

    def maybe_tangent(t):
        assert type(t) is not CCT
        if isinstance(t, torch.Tensor) and t.requires_grad:
            return torch.randn_like(t)
        elif is_tensorlist(t):
            return [torch.randn_like(e) if e.requires_grad else None for e in t]
        return None
    tangent_args = tuple((maybe_tangent(arg) for arg in args))
    flat_kwargs, spec = tree_flatten(kwargs)
    flat_tangent_kwargs = tuple((maybe_tangent(arg) for arg in flat_kwargs))
    tangent_kwargs = tree_unflatten(flat_tangent_kwargs, spec)
    with fwAD.dual_level():

        def maybe_make_dual(dual):
            primal, tangent = dual
            if isinstance(primal, torch.Tensor) and primal.requires_grad:
                return fwAD.make_dual(primal.detach(), tangent)
            elif is_tensorlist(primal):
                return tuple((fwAD.make_dual(pri.detach(), tang) if tang is not None else pri for pri, tang in zip(primal, tangent)))
            return primal

        def compute_expected_grad(args, tangent_args, kwargs, tangent_kwargs):
            op_args = tuple(map(maybe_make_dual, zip(args, tangent_args)))
            op_kwargs = {k: maybe_make_dual((v, tangent_kwargs[k])) for k, v in kwargs.items()}
            if gradcheck_wrapper is None:
                return op(*op_args, **op_kwargs)
            return gradcheck_wrapper(op, *op_args, **op_kwargs)
        expected = compute_expected_grad(args, tangent_args, kwargs, tangent_kwargs)
        expected = tree_map(fwAD.unpack_dual, expected)
        expected_primals = tree_map(lambda x: x.primal, expected)
        expected_tangents = tree_map(lambda x: x.tangent, expected)
        for choice in generate_subclass_choices_args_kwargs(args, kwargs, CCT, cct_mode):
            new_args, new_kwargs, which_args_are_wrapped, which_kwargs_are_wrapped = choice
            for tang_choice in generate_subclass_choices_args_kwargs(tangent_args, tangent_kwargs, CCT, cct_mode):
                new_tang_args, new_tang_kwargs, which_tang_args_are_wrapped, which_tang_kwargs_are_wrapped = tang_choice
                op_args = tuple(map(maybe_make_dual, zip(new_args, new_tang_args)))
                op_kwargs = {k: maybe_make_dual((v, new_tang_kwargs[k])) for k, v in new_kwargs.items()}
                try:
                    if gradcheck_wrapper is None:
                        actual = op(*op_args, **op_kwargs)
                    else:
                        actual = gradcheck_wrapper(op, *op_args, **op_kwargs)
                except RuntimeError as err:
                    raise_composite_compliance_error(err, f'- wrapped_args: {which_args_are_wrapped}\n- wrapped_kwargs: {which_kwargs_are_wrapped}\n- wrapped_tangent_args: {which_tang_args_are_wrapped}\n- wrapped_tangent_kwargs: {which_tang_kwargs_are_wrapped}\n')

                def unwrap(e):
                    return e.elem if isinstance(e, CCT) else e
                actual = tree_map(fwAD.unpack_dual, actual)
                actual_primals = tree_map(lambda x: unwrap(x.primal), actual)
                actual_tangents = tree_map(lambda x: unwrap(x.tangent), actual)
                assert_equal_fn(actual_primals, expected_primals, equal_nan=True)
                assert_equal_fn(actual_tangents, expected_tangents, equal_nan=True)
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
def check_all_permutations(op, args, kwargs, assert_equal_fn):
    CCT, cct_mode = generate_cct_and_mode()
    expected = op(*args, **kwargs)
    for choice in generate_subclass_choices_args_kwargs(args, kwargs, CCT, cct_mode):
        new_args, new_kwargs, which_args_are_wrapped, which_kwargs_are_wrapped = choice
        try:
            actual = op(*new_args, **new_kwargs)
        except RuntimeError as err:
            raise_composite_compliance_error(err, f'- wrapped_args: {which_args_are_wrapped}\n- wrapped_kwargs: {which_kwargs_are_wrapped}\n')

        def unwrap(e):
            return e.elem if isinstance(e, CCT) else e
        assert_equal_fn(tree_map(unwrap, actual), expected)
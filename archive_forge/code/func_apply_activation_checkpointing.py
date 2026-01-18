import warnings
from enum import auto, Enum
from functools import partial
from typing import Any, Callable, Dict, Iterator, Optional, Tuple
import torch
import torch.nn as nn
from torch.autograd.graph import save_on_cpu
from torch.distributed.utils import _pack_kwargs, _replace_by_prefix, _unpack_kwargs
from torch.utils.checkpoint import checkpoint as torch_utils_checkpoint
def apply_activation_checkpointing(model, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=lambda _: True, auto_wrap_policy: Optional[Callable[[nn.Module, bool, int], bool]]=None):
    """
    Apply :func:`checkpoint_wrapper` to modules within `model` based on a user-defined configuration.

    For each module within `model`, the `check_fn` is used to decide
    whether `module` should be wrapped with :func:`checkpoint_wrapper` or not.

    Note::
        This function modifies `model` in place and replaces appropriate layers with
        their checkpoint-wrapped modules.
    Note::
        This function will not wrap the overall root module. If this is needed, please directly use
        :func:`checkpoint_wrapper` or :func:`offload_wrapper`.
    Usage::
        model = nn.Sequential(
            nn.Linear(10, 10), nn.Linear(10, 10), nn.Linear(10, 10)
        )
        check_fn = lambda l: isinstance(l, nn.Linear)
        # checkpoint activations
        apply_activation_checkpointing(model, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=check_fn)
        # Or offload activations to CPU
        apply_activation_checkpointing(model, checkpoint_wrapper_fn=offload_wrapper, check_fn=check_fn)
    Args:
        model (nn.Module):
            The model whose submodules should be wrapped with activation checkpointing.
        checkpoint_wrapper_fn (Optional[Callable[nn.Module]])
            A ``Callable`` which will wrap modules
        check_fn (Optional[Callable[nn.Module, nn.Module]])
            A lambda function which will be passed each child submodule of ``model`` and returns
            ``True`` or ``False`` depending on whether the submodule should be wrapped.
        auto_wrap_policy (Optional[Callable[[nn.Module, bool, int], bool]]): A policy to wrap model's
            submodules with AC. Note that if this is specified, it takes precedence over ``check_fn``.
    Returns: None (`model` is modified inplace)
    """
    from torch.distributed.fsdp.wrap import _recursive_wrap, lambda_auto_wrap_policy, _Policy
    from torch.distributed.fsdp._wrap_utils import _construct_wrap_fn, _post_order_apply
    policy = auto_wrap_policy if auto_wrap_policy is not None else partial(lambda_auto_wrap_policy, lambda_fn=check_fn)
    if not callable(policy):
        if not isinstance(policy, _Policy):
            raise ValueError(f'Expected {policy} to be callable or be a pre-defined wrap policy')
        target_module_to_kwargs = policy._run_policy(model, ignored_modules=set(), root_kwargs={})
        wrap_fn = _construct_wrap_fn(model, target_module_to_kwargs, checkpoint_wrapper_fn)
        _post_order_apply(model, wrap_fn)
        return
    _recursive_wrap(module=model, auto_wrap_policy=policy, wrapper_cls=checkpoint_wrapper_fn, ignored_modules=set(), ignored_params=set(), only_wrap_children=True)
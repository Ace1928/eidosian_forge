import warnings
from enum import auto, Enum
from functools import partial
from typing import Any, Callable, Dict, Iterator, Optional, Tuple
import torch
import torch.nn as nn
from torch.autograd.graph import save_on_cpu
from torch.distributed.utils import _pack_kwargs, _replace_by_prefix, _unpack_kwargs
from torch.utils.checkpoint import checkpoint as torch_utils_checkpoint
def checkpoint_wrapper(module: torch.nn.Module, checkpoint_impl: CheckpointImpl=CheckpointImpl.NO_REENTRANT, checkpoint_fn=None, **checkpoint_fn_kwargs) -> torch.nn.Module:
    """
    Wrap a module for activation checkpointing.

    If the module is wrapped with this function, all subsequent calls to the module will,
    automatically perform checkpointing without the user having to explicitly call ``checkpoint`` function.

    Usage::
        checkpointed_module = checkpoint_wrapper(module)
        outputs = checkpointed_module(inputs)
    Args:
        module (nn.Module):
            The module to be wrapped
        checkpoint_impl (Optional[CheckpointImpl]):
            The checkpointing implementation to use. Note that this will only
            be passed into the ``torch.utils.checkpoint.checkpoint``
            implementation, and is ignored if a custom ``checkpoint_fn`` is
            specified. Note that for implementations using reentrant checkpoint
            from ``torch.utils.checkpoint``, keyword arguments will only be
            supported if ``checkpoint_impl`` is passed as ``CheckpointImpl.REENTRANT`.
        checkpoint_fn (Optional[Callable]):
            Functional checkpoint implementation to use. If this is specified,
            it will be used over the default ``torch.utils.checkpoint.checkpoint``
            implementation and the `checkpoint_impl` argument will be ignored.
        **checkpoint_fn_kwargs: (Dict[str, Any]): Keyword arguments to pass into `checkpoint_fn`.

    Returns:
        (nn.Module):
            Wrapped module
    """
    if checkpoint_impl == CheckpointImpl.REENTRANT:
        warnings.warn(f'Please specify {CheckpointImpl.NO_REENTRANT} as {CheckpointImpl.REENTRANT} will soon be removed as the default and eventually deprecated.', stacklevel=1)
    return CheckpointWrapper(module, checkpoint_impl, checkpoint_fn, **checkpoint_fn_kwargs)
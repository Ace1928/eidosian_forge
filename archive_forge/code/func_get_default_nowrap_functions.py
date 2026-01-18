import __future__  # noqa: F404
import collections
import functools
import types
import warnings
from typing import Dict, Set, List, Any, Callable, Iterable, Type, Tuple
from functools import wraps
import contextlib
import torch
from torch._C import (
@functools.lru_cache(None)
def get_default_nowrap_functions() -> Set[Callable]:
    """
    Return public functions that do not wrap in a subclass when invoked by
    the default ``Tensor.__torch_function__`` that preserves subclasses.  Typically,
    these functions represent field accesses (i.e., retrieving a Tensor that
    is stored somewhere on the Tensor) as opposed to computation.  Users of
    these functions expect object identity to be preserved over multiple accesses
    (e.g., ``a.grad is a.grad``) which cannot be upheld if we're wrapping on
    the fly every time (furthermore, the tensor stored here might already be
    the subclass, in which case wrapping really ought not to happen).

    Not ALL property accessors have this property; for example ``Tensor.T`` actually
    just creates a new transposed tensor on the fly, and so we SHOULD interpose on
    these calls (you need to check the implementation of the function to see if
    this is the case or not).  Additionally, if a property accessor doesn't return a Tensor,
    it doesn't have to be on this list (though it is harmless if it is).
    """
    Tensor = torch.Tensor
    return {Tensor._base.__get__, Tensor.grad.__get__, Tensor._grad.__get__}
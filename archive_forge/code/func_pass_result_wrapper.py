import inspect
import logging
from queue import Queue
from functools import wraps
from typing import Callable, Dict, List
import torch.nn as nn
from torch.fx.graph_module import GraphModule
from torch.fx._compatibility import compatibility
from torch.fx.passes.infra.pass_base import PassResult
@compatibility(is_backward_compatible=False)
def pass_result_wrapper(fn: Callable) -> Callable:
    """
    Wrapper for passes which currently do not return a PassResult.
    This wrapper makes them return a PassResult containing the modified object
    and True for the "modified" flag.

    Args:
        fn (Callable[Module, Any])

    Returns:
        wrapped_fn (Callable[Module, PassResult])
    """
    if fn is None:
        return None

    @wraps(fn)
    def wrapped_fn(gm):
        res = fn(gm)
        if res is None:
            return PassResult(gm, True)
        if isinstance(res, PassResult):
            return res
        elif isinstance(res, nn.Module):
            return PassResult(res, True)
    if not inspect.isfunction(fn):
        wrapped_fn.__name__ = type(fn).__name__
    return wrapped_fn
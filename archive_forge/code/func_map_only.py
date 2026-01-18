import functools
import warnings
from typing import (
import torch
import optree
from optree import PyTreeSpec  # direct import for type annotations
def map_only(__type_or_types: TypeAny) -> MapOnlyFn[FnAny[Any]]:
    """
    Suppose you are writing a tree_map over tensors, leaving everything
    else unchanged.  Ordinarily you would have to write:

        def go(t):
            if isinstance(t, Tensor):
                return ...
            else:
                return t

    With this function, you only need to write:

        @map_only(Tensor)
        def go(t):
            return ...

    You can also directly use 'tree_map_only'
    """

    def wrapper(func: Callable[[T], Any]) -> Callable[[Any], Any]:

        @functools.wraps(func)
        def wrapped(x: T) -> Any:
            if isinstance(x, __type_or_types):
                return func(x)
            return x
        return wrapped
    return wrapper
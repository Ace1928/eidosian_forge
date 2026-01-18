from git.util import (
from .symbolic import SymbolicReference, T_References
from typing import Any, Callable, Iterator, Type, Union, TYPE_CHECKING
from git.types import Commit_ish, PathLike, _T
def require_remote_ref_path(func: Callable[..., _T]) -> Callable[..., _T]:
    """A decorator raising a TypeError if we are not a valid remote, based on the path."""

    def wrapper(self: T_References, *args: Any) -> _T:
        if not self.is_remote():
            raise ValueError('ref path does not point to a remote reference: %s' % self.path)
        return func(self, *args)
    wrapper.__name__ = func.__name__
    return wrapper
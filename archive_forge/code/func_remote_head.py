from git.util import (
from .symbolic import SymbolicReference, T_References
from typing import Any, Callable, Iterator, Type, Union, TYPE_CHECKING
from git.types import Commit_ish, PathLike, _T
@property
@require_remote_ref_path
def remote_head(self) -> str:
    """
        :return: Name of the remote head itself, e.g. master.

        :note: The returned name is usually not qualified enough to uniquely identify
            a branch.
        """
    tokens = self.path.split('/')
    return '/'.join(tokens[3:])
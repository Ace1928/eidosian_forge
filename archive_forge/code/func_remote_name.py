from git.util import (
from .symbolic import SymbolicReference, T_References
from typing import Any, Callable, Iterator, Type, Union, TYPE_CHECKING
from git.types import Commit_ish, PathLike, _T
@property
@require_remote_ref_path
def remote_name(self) -> str:
    """
        :return:
            Name of the remote we are a reference of, such as 'origin' for a reference
            named 'origin/master'.
        """
    tokens = self.path.split('/')
    return tokens[2]
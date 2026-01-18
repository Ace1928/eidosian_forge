from git.util import IterableList, join_path
import git.diff as git_diff
from git.util import to_bin_sha
from . import util
from .base import IndexObject, IndexObjUnion
from .blob import Blob
from .submodule.base import Submodule
from .fun import tree_entries_from_data, tree_to_stream
from typing import (
from git.types import PathLike, Literal
def add_unchecked(self, binsha: bytes, mode: int, name: str) -> None:
    """Add the given item to the tree. Its correctness is assumed, so it is the
        caller's responsibility to ensure that the input is correct.

        For more information on the parameters, see :meth:`add`.

        :param binsha: 20 byte binary sha
        """
    assert isinstance(binsha, bytes) and isinstance(mode, int) and isinstance(name, str)
    tree_cache = (binsha, mode, name)
    self._cache.append(tree_cache)
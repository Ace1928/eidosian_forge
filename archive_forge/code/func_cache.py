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
@property
def cache(self) -> TreeModifier:
    """
        :return: An object allowing to modify the internal cache. This can be used
            to change the tree's contents. When done, make sure you call ``set_done``
            on the tree modifier, or serialization behaviour will be incorrect.
            See :class:`TreeModifier` for more information on how to alter the cache.
        """
    return TreeModifier(self._cache)
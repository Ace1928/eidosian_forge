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
def set_done(self) -> 'TreeModifier':
    """Call this method once you are done modifying the tree information.

        This may be called several times, but be aware that each call will cause
        a sort operation.

        :return self:
        """
    self._cache.sort(key=lambda x: x[2] + '/' if x[1] == Tree.tree_id << 12 else x[2])
    return self
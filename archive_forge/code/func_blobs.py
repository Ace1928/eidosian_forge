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
def blobs(self) -> List[Blob]:
    """:return: list(Blob, ...) list of blobs directly below this tree"""
    return [i for i in self if i.type == 'blob']
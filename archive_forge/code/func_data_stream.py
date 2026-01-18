from git.exc import WorkTreeRepositoryUnsupported
from git.util import LazyMixin, join_path_native, stream_copy, bin_to_hex
import gitdb.typ as dbtyp
import os.path as osp
from .util import get_object_type_by_name
from typing import Any, TYPE_CHECKING, Union
from git.types import PathLike, Commit_ish, Lit_commit_ish
@property
def data_stream(self) -> 'OStream':
    """
        :return: File Object compatible stream to the uncompressed raw data of the object

        :note: Returned streams must be read in order.
        """
    return self.repo.odb.stream(self.binsha)
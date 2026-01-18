from git.exc import WorkTreeRepositoryUnsupported
from git.util import LazyMixin, join_path_native, stream_copy, bin_to_hex
import gitdb.typ as dbtyp
import os.path as osp
from .util import get_object_type_by_name
from typing import Any, TYPE_CHECKING, Union
from git.types import PathLike, Commit_ish, Lit_commit_ish
def stream_data(self, ostream: 'OStream') -> 'Object':
    """Write our data directly to the given output stream.

        :param ostream: File object compatible stream object.
        :return: self
        """
    istream = self.repo.odb.stream(self.binsha)
    stream_copy(istream, ostream)
    return self
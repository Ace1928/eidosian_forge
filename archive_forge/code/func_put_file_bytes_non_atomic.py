import os
import posixpath
import stat
from dulwich.index import index_entry_from_stat
from dulwich.objects import Blob, Tree
from breezy import errors, lock, osutils
from breezy import revision as _mod_revision
from breezy import tree as _mod_tree
from breezy import urlutils
from breezy.transport.memory import MemoryTransport
from .mapping import decode_git_path, encode_git_path
from .tree import MutableGitIndexTree
def put_file_bytes_non_atomic(self, path, bytes):
    """See MutableTree.put_file_bytes_non_atomic."""
    self._file_transport.put_bytes(path, bytes)
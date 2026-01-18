import os
from io import BytesIO
from ..lazy_import import lazy_import
import contextlib
import errno
import stat
from breezy import (
from breezy.bzr import (
from .. import errors
from .. import revision as _mod_revision
from ..lock import LogicalLockResult
from ..lockable_files import LockableFiles
from ..lockdir import LockDir
from ..mutabletree import BadReferenceTarget, MutableTree
from ..osutils import file_kind, isdir, pathjoin, realpath, safe_unicode
from ..transport import NoSuchFile, get_transport_from_path
from ..transport.local import LocalTransport
from ..tree import FileTimestampUnavailable, InterTree, MissingNestedTree
from ..workingtree import WorkingTree
from . import dirstate
from .inventory import ROOT_ID, Inventory, entry_factory
from .inventorytree import (InterInventoryTree, InventoryRevisionTree,
from .workingtree import InventoryWorkingTree, WorkingTreeFormatMetaDir
def move_one(old_entry, from_path_utf8, minikind, executable, fingerprint, packed_stat, size, to_block, to_key, to_path_utf8):
    state._make_absent(old_entry)
    from_key = old_entry[0]
    rollbacks.callback(state.update_minimal, from_key, minikind, executable=executable, fingerprint=fingerprint, packed_stat=packed_stat, size=size, path_utf8=from_path_utf8)
    state.update_minimal(to_key, minikind, executable=executable, fingerprint=fingerprint, packed_stat=packed_stat, size=size, path_utf8=to_path_utf8)
    added_entry_index, _ = state._find_entry_index(to_key, to_block[1])
    new_entry = to_block[1][added_entry_index]
    rollbacks.callback(state._make_absent, new_entry)
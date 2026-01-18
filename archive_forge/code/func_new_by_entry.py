import contextlib
import errno
import os
import time
from stat import S_IEXEC, S_ISREG
from typing import Callable
from . import config as _mod_config
from . import controldir, errors, lazy_import, lock, osutils, registry, trace
from breezy import (
from breezy.i18n import gettext
from .errors import BzrError, DuplicateKey, InternalBzrError
from .filters import ContentFilterContext, filtered_output_bytes
from .mutabletree import MutableTree
from .osutils import delete_any, file_kind, pathjoin, sha_file, splitpath
from .progress import ProgressPhase
from .transport import FileExists, NoSuchFile
from .tree import InterTree, find_previous_path
def new_by_entry(path, tt, entry, parent_id, tree):
    """Create a new file according to its inventory entry"""
    name = entry.name
    kind = entry.kind
    if kind == 'file':
        with tree.get_file(path) as f:
            executable = tree.is_executable(path)
            return tt.new_file(name, parent_id, osutils.file_iterator(f), entry.file_id, executable)
    elif kind in ('directory', 'tree-reference'):
        trans_id = tt.new_directory(name, parent_id, entry.file_id)
        if kind == 'tree-reference':
            tt.set_tree_reference(entry.reference_revision, trans_id)
        return trans_id
    elif kind == 'symlink':
        target = tree.get_symlink_target(path)
        return tt.new_symlink(name, parent_id, target, entry.file_id)
    else:
        raise errors.BadFileKindError(name, kind)
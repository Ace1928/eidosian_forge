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
def resolve_non_directory_parent(tt, path_tree, c_type, parent_id):
    parent_parent = tt.final_parent(parent_id)
    parent_name = tt.final_name(parent_id)
    if tt._tree.supports_setting_file_ids():
        parent_file_id = tt.final_file_id(parent_id)
    else:
        parent_file_id = b'DUMMY'
    new_parent_id = tt.new_directory(parent_name + '.new', parent_parent, parent_file_id)
    _reparent_transform_children(tt, parent_id, new_parent_id)
    if parent_file_id is not None:
        tt.unversion_file(parent_id)
    yield (c_type, 'Created directory', new_parent_id)
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
def link_tree(target_tree, source_tree):
    """Where possible, hard-link files in a tree to those in another tree.

    :param target_tree: Tree to change
    :param source_tree: Tree to hard-link from
    """
    with target_tree.transform() as tt:
        for change in target_tree.iter_changes(source_tree, include_unchanged=True):
            if change.changed_content:
                continue
            if change.kind != ('file', 'file'):
                continue
            if change.executable[0] != change.executable[1]:
                continue
            trans_id = tt.trans_id_tree_path(change.path[1])
            tt.delete_contents(trans_id)
            tt.create_hardlink(source_tree.abspath(change.path[0]), trans_id)
        tt.apply()
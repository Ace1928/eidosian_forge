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
def rollback(self):
    """Reverse all renames that have been performed"""
    for from_, to in reversed(self.past_renames):
        try:
            os.rename(to, from_)
        except OSError as e:
            raise TransformRenameFailed(to, from_, str(e), e.errno)
    self.past_renames = None
    self.pending_deletions = None
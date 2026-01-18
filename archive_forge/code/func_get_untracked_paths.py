import datetime
import fnmatch
import os
import posixpath
import stat
import sys
import time
from collections import namedtuple
from contextlib import closing, contextmanager
from io import BytesIO, RawIOBase
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from .archive import tar_stream
from .client import get_transport_and_path
from .config import Config, ConfigFile, StackedConfig, read_submodules
from .diff_tree import (
from .errors import SendPackError
from .file import ensure_dir_exists
from .graph import can_fast_forward
from .ignore import IgnoreFilterManager
from .index import (
from .object_store import iter_tree_contents, tree_lookup_path
from .objects import (
from .objectspec import (
from .pack import write_pack_from_container, write_pack_index
from .patch import write_tree_diff
from .protocol import ZERO_SHA, Protocol
from .refs import (
from .repo import BaseRepo, Repo
from .server import (
from .server import update_server_info as server_update_server_info
def get_untracked_paths(frompath, basepath, index, exclude_ignored=False, untracked_files='all'):
    """Get untracked paths.

    Args:
      frompath: Path to walk
      basepath: Path to compare to
      index: Index to check against
      exclude_ignored: Whether to exclude ignored paths
      untracked_files: How to handle untracked files:
        - "no": return an empty list
        - "all": return all files in untracked directories
        - "normal": Not implemented

    Note: ignored directories will never be walked for performance reasons.
      If exclude_ignored is False, only the path to an ignored directory will
      be yielded, no files inside the directory will be returned
    """
    if untracked_files == 'normal':
        raise NotImplementedError('normal is not yet supported')
    if untracked_files not in ('no', 'all'):
        raise ValueError('untracked_files must be one of (no, all)')
    if untracked_files == 'no':
        return
    with open_repo_closing(basepath) as r:
        ignore_manager = IgnoreFilterManager.from_repo(r)
    ignored_dirs = []

    def prune_dirnames(dirpath, dirnames):
        for i in range(len(dirnames) - 1, -1, -1):
            path = os.path.join(dirpath, dirnames[i])
            ip = os.path.join(os.path.relpath(path, basepath), '')
            if ignore_manager.is_ignored(ip):
                if not exclude_ignored:
                    ignored_dirs.append(os.path.join(os.path.relpath(path, frompath), ''))
                del dirnames[i]
        return dirnames
    for ap, is_dir in _walk_working_dir_paths(frompath, basepath, prune_dirnames=prune_dirnames):
        if not is_dir:
            ip = path_to_tree_path(basepath, ap)
            if ip not in index:
                if not exclude_ignored or not ignore_manager.is_ignored(os.path.relpath(ap, basepath)):
                    yield os.path.relpath(ap, frompath)
    yield from ignored_dirs
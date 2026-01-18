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
def checkout_branch(repo, target: Union[bytes, str], force: bool=False):
    """Switch branches or restore working tree files.

    The implementation of this function will probably not scale well
    for branches with lots of local changes.
    This is due to the analysis of a diff between branches before any
    changes are applied.

    Args:
      repo: dulwich Repo object
      target: branch name or commit sha to checkout
      force: true or not to force checkout
    """
    target = to_bytes(target)
    current_tree = parse_tree(repo, repo.head())
    target_tree = parse_tree(repo, target)
    if force:
        repo.reset_index(target_tree.id)
        _update_head_during_checkout_branch(repo, target)
    else:
        status_report = status(repo)
        changes = list(set(status_report[0]['add'] + status_report[0]['delete'] + status_report[0]['modify'] + status_report[1]))
        index = 0
        while index < len(changes):
            change = changes[index]
            try:
                current_tree.lookup_path(repo.object_store.__getitem__, change)
                try:
                    target_tree.lookup_path(repo.object_store.__getitem__, change)
                    index += 1
                except KeyError:
                    raise CheckoutError('Your local changes to the following files would be overwritten by checkout: ' + change.decode())
            except KeyError:
                changes.pop(index)
        checkout_target = _update_head_during_checkout_branch(repo, target)
        if checkout_target is not None:
            target_tree = parse_tree(repo, checkout_target)
        dealt_with = set()
        repo_index = repo.open_index()
        for entry in iter_tree_contents(repo.object_store, target_tree.id):
            dealt_with.add(entry.path)
            if entry.path in changes:
                continue
            full_path = os.path.join(os.fsencode(repo.path), entry.path)
            blob = repo.object_store[entry.sha]
            ensure_dir_exists(os.path.dirname(full_path))
            st = build_file_from_blob(blob, entry.mode, full_path)
            repo_index[entry.path] = index_entry_from_stat(st, entry.sha)
        repo_index.write()
        for entry in iter_tree_contents(repo.object_store, current_tree.id):
            if entry.path not in dealt_with:
                repo.unstage([entry.path])
    repo_index = repo.open_index()
    for change in repo_index.changes_from_tree(repo.object_store, current_tree.id):
        path_change = change[0]
        if path_change[1] is None:
            file_name = path_change[0]
            full_path = os.path.join(repo.path, file_name.decode())
            if os.path.isfile(full_path):
                os.remove(full_path)
            dir_path = os.path.dirname(full_path)
            while dir_path != repo.path:
                is_empty = len(os.listdir(dir_path)) == 0
                if is_empty:
                    os.rmdir(dir_path)
                dir_path = os.path.dirname(dir_path)
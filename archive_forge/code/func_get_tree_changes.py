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
def get_tree_changes(repo):
    """Return add/delete/modify changes to tree by comparing index to HEAD.

    Args:
      repo: repo path or object
    Returns: dict with lists for each type of change
    """
    with open_repo_closing(repo) as r:
        index = r.open_index()
        tracked_changes = {'add': [], 'delete': [], 'modify': []}
        try:
            tree_id = r[b'HEAD'].tree
        except KeyError:
            tree_id = None
        for change in index.changes_from_tree(r.object_store, tree_id):
            if not change[0][0]:
                tracked_changes['add'].append(change[0][1])
            elif not change[0][1]:
                tracked_changes['delete'].append(change[0][0])
            elif change[0][0] == change[0][1]:
                tracked_changes['modify'].append(change[0][0])
            else:
                raise NotImplementedError('git mv ops not yet supported')
        return tracked_changes
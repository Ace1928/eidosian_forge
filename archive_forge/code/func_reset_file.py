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
def reset_file(repo, file_path: str, target: bytes=b'HEAD', symlink_fn=None):
    """Reset the file to specific commit or branch.

    Args:
      repo: dulwich Repo object
      file_path: file to reset, relative to the repository path
      target: branch or commit or b'HEAD' to reset
    """
    tree = parse_tree(repo, treeish=target)
    tree_path = _fs_to_tree_path(file_path)
    file_entry = tree.lookup_path(repo.object_store.__getitem__, tree_path)
    full_path = os.path.join(os.fsencode(repo.path), tree_path)
    blob = repo.object_store[file_entry[1]]
    mode = file_entry[0]
    build_file_from_blob(blob, mode, full_path, symlink_fn=symlink_fn)
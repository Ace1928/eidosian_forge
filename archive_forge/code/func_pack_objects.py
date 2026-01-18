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
def pack_objects(repo, object_ids, packf, idxf, delta_window_size=None, deltify=None, reuse_deltas=True):
    """Pack objects into a file.

    Args:
      repo: Path to the repository
      object_ids: List of object ids to write
      packf: File-like object to write to
      idxf: File-like object to write to (can be None)
      delta_window_size: Sliding window size for searching for deltas;
                         Set to None for default window size.
      deltify: Whether to deltify objects
      reuse_deltas: Allow reuse of existing deltas while deltifying
    """
    with open_repo_closing(repo) as r:
        entries, data_sum = write_pack_from_container(packf.write, r.object_store, [(oid, None) for oid in object_ids], deltify=deltify, delta_window_size=delta_window_size, reuse_deltas=reuse_deltas)
    if idxf is not None:
        entries = sorted([(k, v[0], v[1]) for k, v in entries.items()])
        write_pack_index(idxf, entries, data_sum)
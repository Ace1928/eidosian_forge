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
def path_to_tree_path(repopath, path, tree_encoding=DEFAULT_ENCODING):
    """Convert a path to a path usable in an index, e.g. bytes and relative to
    the repository root.

    Args:
      repopath: Repository path, absolute or relative to the cwd
      path: A path, absolute or relative to the cwd
    Returns: A path formatted for use in e.g. an index
    """
    if sys.platform == 'win32':
        path = os.path.abspath(path)
    path = Path(path)
    resolved_path = path.resolve()
    if sys.platform == 'win32':
        repopath = os.path.abspath(repopath)
    repopath = Path(repopath).resolve()
    try:
        relpath = resolved_path.relative_to(repopath)
    except ValueError:
        if path.is_symlink():
            parent = path.parent.resolve()
            relpath = (parent / path.name).relative_to(repopath)
        else:
            raise
    if sys.platform == 'win32':
        return str(relpath).replace(os.path.sep, '/').encode(tree_encoding)
    else:
        return bytes(relpath)
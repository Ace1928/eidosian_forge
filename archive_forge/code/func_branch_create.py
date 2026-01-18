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
def branch_create(repo, name, objectish=None, force=False):
    """Create a branch.

    Args:
      repo: Path to the repository
      name: Name of the new branch
      objectish: Target object to point new branch at (defaults to HEAD)
      force: Force creation of branch, even if it already exists
    """
    with open_repo_closing(repo) as r:
        if objectish is None:
            objectish = 'HEAD'
        object = parse_object(r, objectish)
        refname = _make_branch_ref(name)
        ref_message = b'branch: Created from ' + objectish.encode(DEFAULT_ENCODING)
        if force:
            r.refs.set_if_equals(refname, None, object.id, message=ref_message)
        elif not r.refs.add_if_new(refname, object.id, message=ref_message):
            raise Error('Branch with name %s already exists.' % name)
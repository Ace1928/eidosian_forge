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
def get_user_timezones():
    """Retrieve local timezone as described in
    https://raw.githubusercontent.com/git/git/v2.3.0/Documentation/date-formats.txt
    Returns: A tuple containing author timezone, committer timezone.
    """
    local_timezone = time.localtime().tm_gmtoff
    if os.environ.get('GIT_AUTHOR_DATE'):
        author_timezone = parse_timezone_format(os.environ['GIT_AUTHOR_DATE'])
    else:
        author_timezone = local_timezone
    if os.environ.get('GIT_COMMITTER_DATE'):
        commit_timezone = parse_timezone_format(os.environ['GIT_COMMITTER_DATE'])
    else:
        commit_timezone = local_timezone
    return (author_timezone, commit_timezone)
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
def tag_create(repo, tag, author=None, message=None, annotated=False, objectish='HEAD', tag_time=None, tag_timezone=None, sign=False, encoding=DEFAULT_ENCODING):
    """Creates a tag in git via dulwich calls.

    Args:
      repo: Path to repository
      tag: tag string
      author: tag author (optional, if annotated is set)
      message: tag message (optional)
      annotated: whether to create an annotated tag
      objectish: object the tag should point at, defaults to HEAD
      tag_time: Optional time for annotated tag
      tag_timezone: Optional timezone for annotated tag
      sign: GPG Sign the tag (bool, defaults to False,
        pass True to use default GPG key,
        pass a str containing Key ID to use a specific GPG key)
    """
    with open_repo_closing(repo) as r:
        object = parse_object(r, objectish)
        if annotated:
            tag_obj = Tag()
            if author is None:
                author = r._get_user_identity(r.get_config_stack())
            tag_obj.tagger = author
            tag_obj.message = message + '\n'.encode(encoding)
            tag_obj.name = tag
            tag_obj.object = (type(object), object.id)
            if tag_time is None:
                tag_time = int(time.time())
            tag_obj.tag_time = tag_time
            if tag_timezone is None:
                tag_timezone = get_user_timezones()[1]
            elif isinstance(tag_timezone, str):
                tag_timezone = parse_timezone(tag_timezone)
            tag_obj.tag_timezone = tag_timezone
            if sign:
                tag_obj.sign(sign if isinstance(sign, str) else None)
            r.object_store.add_object(tag_obj)
            tag_id = tag_obj.id
        else:
            tag_id = object.id
        r.refs[_make_tag_ref(tag)] = tag_id
from __future__ import annotations
import gc
import logging
import os
import os.path as osp
from pathlib import Path
import re
import shlex
import warnings
import gitdb
from gitdb.db.loose import LooseObjectDB
from gitdb.exc import BadObject
from git.cmd import Git, handle_process_output
from git.compat import defenc, safe_decode
from git.config import GitConfigParser
from git.db import GitCmdObjectDB
from git.exc import (
from git.index import IndexFile
from git.objects import Submodule, RootModule, Commit
from git.refs import HEAD, Head, Reference, TagReference
from git.remote import Remote, add_progress, to_progress_instance
from git.util import (
from .fun import (
from git.types import (
from typing import (
from git.types import ConfigLevels_Tup, TypedDict
def is_valid_object(self, sha: str, object_type: Union[str, None]=None) -> bool:
    try:
        complete_sha = self.odb.partial_to_complete_sha_hex(sha)
        object_info = self.odb.info(complete_sha)
        if object_type:
            if object_info.type == object_type.encode():
                return True
            else:
                _logger.debug("Commit hash points to an object of type '%s'. Requested were objects of type '%s'", object_info.type.decode(), object_type)
                return False
        else:
            return True
    except BadObject:
        _logger.debug('Commit hash is invalid.')
        return False
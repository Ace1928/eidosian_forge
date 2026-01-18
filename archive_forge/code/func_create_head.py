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
def create_head(self, path: PathLike, commit: Union['SymbolicReference', 'str']='HEAD', force: bool=False, logmsg: Optional[str]=None) -> 'Head':
    """Create a new head within the repository.

        :note: For more documentation, please see the
            :meth:`Head.create <git.refs.head.Head.create>` method.

        :return: Newly created :class:`~git.refs.head.Head` Reference
        """
    return Head.create(self, path, commit, logmsg, force)
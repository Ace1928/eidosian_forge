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
def ignored(self, *paths: PathLike) -> List[str]:
    """Checks if paths are ignored via .gitignore.

        This does so using the ``git check-ignore`` method.

        :param paths: List of paths to check whether they are ignored or not

        :return: Subset of those paths which are ignored
        """
    try:
        proc: str = self.git.check_ignore(*paths)
    except GitCommandError as err:
        if err.status == 1:
            return []
        else:
            raise
    return proc.replace('\\\\', '\\').replace('"', '').split('\n')
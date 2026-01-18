import os
from git.compat import defenc
from git.objects import Object
from git.objects.commit import Commit
from git.util import (
from gitdb.exc import BadObject, BadName
from .log import RefLog
from typing import (
from git.types import Commit_ish, PathLike
def is_remote(self) -> bool:
    """:return: True if this symbolic reference points to a remote branch"""
    return str(self.path).startswith(self._remote_common_path_default + '/')
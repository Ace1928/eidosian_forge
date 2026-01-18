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
def merge_base(self, *rev: TBD, **kwargs: Any) -> List[Union[Commit_ish, None]]:
    """Find the closest common ancestor for the given revision (Commits, Tags, References, etc.).

        :param rev: At least two revs to find the common ancestor for.
        :param kwargs: Additional arguments to be passed to the
            ``repo.git.merge_base()`` command which does all the work.
        :return: A list of :class:`~git.objects.commit.Commit` objects. If ``--all`` was
            not passed as a keyword argument, the list will have at max one
            :class:`~git.objects.commit.Commit`, or is empty if no common merge base
            exists.
        :raises ValueError: If not at least two revs are provided.
        """
    if len(rev) < 2:
        raise ValueError('Please specify at least two revs, got only %i' % len(rev))
    res: List[Union[Commit_ish, None]] = []
    try:
        lines = self.git.merge_base(*rev, **kwargs).splitlines()
    except GitCommandError as err:
        if err.status == 128:
            raise
        return res
    for line in lines:
        res.append(self.commit(line))
    return res
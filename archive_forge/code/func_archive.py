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
def archive(self, ostream: Union[TextIO, BinaryIO], treeish: Optional[str]=None, prefix: Optional[str]=None, **kwargs: Any) -> Repo:
    """Archive the tree at the given revision.

        :param ostream: file compatible stream object to which the archive will be written as bytes.

        :param treeish: is the treeish name/id, defaults to active branch.

        :param prefix: is the optional prefix to prepend to each filename in the archive.

        :param kwargs: Additional arguments passed to git-archive:

            * Use the 'format' argument to define the kind of format. Use
              specialized ostreams to write any format supported by python.
            * You may specify the special **path** keyword, which may either be a repository-relative
              path to a directory or file to place into the archive, or a list or tuple of multiple paths.

        :raise GitCommandError: If something went wrong.

        :return: self
        """
    if treeish is None:
        treeish = self.head.commit
    if prefix and 'prefix' not in kwargs:
        kwargs['prefix'] = prefix
    kwargs['output_stream'] = ostream
    path = kwargs.pop('path', [])
    path = cast(Union[PathLike, List[PathLike], Tuple[PathLike, ...]], path)
    if not isinstance(path, (tuple, list)):
        path = [path]
    self.git.archive('--', treeish, *path, **kwargs)
    return self
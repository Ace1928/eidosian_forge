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
@classmethod
def clone_from(cls, url: PathLike, to_path: PathLike, progress: CallableProgress=None, env: Optional[Mapping[str, str]]=None, multi_options: Optional[List[str]]=None, allow_unsafe_protocols: bool=False, allow_unsafe_options: bool=False, **kwargs: Any) -> 'Repo':
    """Create a clone from the given URL.

        :param url: Valid git url, see http://www.kernel.org/pub/software/scm/git/docs/git-clone.html#URLS

        :param to_path: Path to which the repository should be cloned to.

        :param progress: See :meth:`git.remote.Remote.push`.

        :param env: Optional dictionary containing the desired environment variables.

            Note: Provided variables will be used to update the execution
            environment for `git`. If some variable is not specified in `env`
            and is defined in `os.environ`, value from `os.environ` will be used.
            If you want to unset some variable, consider providing empty string
            as its value.

        :param multi_options: See :meth:`clone` method.

        :param allow_unsafe_protocols: Allow unsafe protocols to be used, like ext.

        :param allow_unsafe_options: Allow unsafe options to be used, like --upload-pack.

        :param kwargs: See the :meth:`clone` method.

        :return: :class:`Repo` instance pointing to the cloned directory.
        """
    git = cls.GitCommandWrapperType(os.getcwd())
    if env is not None:
        git.update_environment(**env)
    return cls._clone(git, url, to_path, GitCmdObjectDB, progress, multi_options, allow_unsafe_protocols=allow_unsafe_protocols, allow_unsafe_options=allow_unsafe_options, **kwargs)
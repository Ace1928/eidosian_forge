import gc
from io import BytesIO
import logging
import os
import os.path as osp
import stat
import uuid
import git
from git.cmd import Git
from git.compat import defenc
from git.config import GitConfigParser, SectionConstraint, cp
from git.exc import (
from git.objects.base import IndexObject, Object
from git.objects.util import TraversableIterableObj
from git.util import (
from .util import (
from typing import Callable, Dict, Mapping, Sequence, TYPE_CHECKING, cast
from typing import Any, Iterator, Union
from git.types import Commit_ish, Literal, PathLike, TBD
@unbare_repo
def module(self) -> 'Repo':
    """
        :return: Repo instance initialized from the repository at our submodule path

        :raise InvalidGitRepositoryError: If a repository was not available. This could
            also mean that it was not yet initialized.
        """
    module_checkout_abspath = self.abspath
    try:
        repo = git.Repo(module_checkout_abspath)
        if repo != self.repo:
            return repo
    except (InvalidGitRepositoryError, NoSuchPathError) as e:
        raise InvalidGitRepositoryError('No valid repository at %s' % module_checkout_abspath) from e
    else:
        raise InvalidGitRepositoryError('Repository at %r was not yet checked out' % module_checkout_abspath)
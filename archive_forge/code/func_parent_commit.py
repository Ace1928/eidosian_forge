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
@property
def parent_commit(self) -> 'Commit_ish':
    """
        :return: Commit instance with the tree containing the .gitmodules file

        :note: Will always point to the current head's commit if it was not set explicitly.
        """
    if self._parent_commit is None:
        return self.repo.commit()
    return self._parent_commit
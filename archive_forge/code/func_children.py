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
def children(self) -> IterableList['Submodule']:
    """
        :return: IterableList(Submodule, ...) an iterable list of submodules instances
            which are children of this submodule or 0 if the submodule is not checked out.
        """
    return self._get_intermediate_items(self)
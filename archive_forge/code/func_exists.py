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
def exists(self) -> bool:
    """
        :return: True if the submodule exists, False otherwise. Please note that
            a submodule may exist (in the .gitmodules file) even though its module
            doesn't exist on disk.
        """
    loc = locals()
    for attr in self._cache_attrs:
        try:
            if hasattr(self, attr):
                loc[attr] = getattr(self, attr)
        except (cp.NoSectionError, ValueError):
            pass
    self._clear_cache()
    try:
        try:
            self.path
            return True
        except Exception:
            return False
    finally:
        for attr in self._cache_attrs:
            if attr in loc:
                setattr(self, attr, loc[attr])
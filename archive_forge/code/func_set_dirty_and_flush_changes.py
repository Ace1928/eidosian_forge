import abc
import configparser as cp
import fnmatch
from functools import wraps
import inspect
from io import BufferedReader, IOBase
import logging
import os
import os.path as osp
import re
import sys
from git.compat import defenc, force_text
from git.util import LockFile
from typing import (
from git.types import Lit_config_levels, ConfigLevels_Tup, PathLike, assert_never, _T
def set_dirty_and_flush_changes(non_const_func: Callable[..., _T]) -> Callable[..., _T]:
    """Return a method that checks whether given non constant function may be called.

    If so, the instance will be set dirty. Additionally, we flush the changes right to disk.
    """

    def flush_changes(self: 'GitConfigParser', *args: Any, **kwargs: Any) -> _T:
        rval = non_const_func(self, *args, **kwargs)
        self._dirty = True
        self.write()
        return rval
    flush_changes.__name__ = non_const_func.__name__
    return flush_changes
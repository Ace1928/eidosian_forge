import contextlib
from functools import wraps
import os
import os.path as osp
import struct
import tempfile
from types import TracebackType
from typing import Any, Callable, TYPE_CHECKING, Optional, Type
from git.types import Literal, PathLike, _T
@wraps(func)
def set_git_working_dir(self: 'IndexFile', *args: Any, **kwargs: Any) -> _T:
    cur_wd = os.getcwd()
    os.chdir(str(self.repo.working_tree_dir))
    try:
        return func(self, *args, **kwargs)
    finally:
        os.chdir(cur_wd)
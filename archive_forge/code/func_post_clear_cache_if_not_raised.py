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
def post_clear_cache_if_not_raised(self: 'IndexFile', *args: Any, **kwargs: Any) -> _T:
    rval = func(self, *args, **kwargs)
    self._delete_entries_cache()
    return rval
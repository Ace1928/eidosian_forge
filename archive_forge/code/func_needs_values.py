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
def needs_values(func: Callable[..., _T]) -> Callable[..., _T]:
    """Return a method for ensuring we read values (on demand) before we try to access them."""

    @wraps(func)
    def assure_data_present(self: 'GitConfigParser', *args: Any, **kwargs: Any) -> _T:
        self.read()
        return func(self, *args, **kwargs)
    return assure_data_present
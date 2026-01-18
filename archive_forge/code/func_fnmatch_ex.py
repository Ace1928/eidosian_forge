import atexit
import contextlib
from enum import Enum
from errno import EBADF
from errno import ELOOP
from errno import ENOENT
from errno import ENOTDIR
import fnmatch
from functools import partial
import importlib.util
import itertools
import os
from os.path import expanduser
from os.path import expandvars
from os.path import isabs
from os.path import sep
from pathlib import Path
from pathlib import PurePath
from posixpath import sep as posix_sep
import shutil
import sys
import types
from types import ModuleType
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
import uuid
import warnings
from _pytest.compat import assert_never
from _pytest.outcomes import skip
from _pytest.warning_types import PytestWarning
def fnmatch_ex(pattern: str, path: Union[str, 'os.PathLike[str]']) -> bool:
    """A port of FNMatcher from py.path.common which works with PurePath() instances.

    The difference between this algorithm and PurePath.match() is that the
    latter matches "**" glob expressions for each part of the path, while
    this algorithm uses the whole path instead.

    For example:
        "tests/foo/bar/doc/test_foo.py" matches pattern "tests/**/doc/test*.py"
        with this algorithm, but not with PurePath.match().

    This algorithm was ported to keep backward-compatibility with existing
    settings which assume paths match according this logic.

    References:
    * https://bugs.python.org/issue29249
    * https://bugs.python.org/issue34731
    """
    path = PurePath(path)
    iswin32 = sys.platform.startswith('win')
    if iswin32 and sep not in pattern and (posix_sep in pattern):
        pattern = pattern.replace(posix_sep, sep)
    if sep not in pattern:
        name = path.name
    else:
        name = str(path)
        if path.is_absolute() and (not os.path.isabs(pattern)):
            pattern = f'*{os.sep}{pattern}'
    return fnmatch.fnmatch(name, pattern)
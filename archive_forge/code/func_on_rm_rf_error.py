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
def on_rm_rf_error(func, path: str, excinfo: Union[BaseException, Tuple[Type[BaseException], BaseException, Optional[types.TracebackType]]], *, start_path: Path) -> bool:
    """Handle known read-only errors during rmtree.

    The returned value is used only by our own tests.
    """
    if isinstance(excinfo, BaseException):
        exc = excinfo
    else:
        exc = excinfo[1]
    if isinstance(exc, FileNotFoundError):
        return False
    if not isinstance(exc, PermissionError):
        warnings.warn(PytestWarning(f'(rm_rf) error removing {path}\n{type(exc)}: {exc}'))
        return False
    if func not in (os.rmdir, os.remove, os.unlink):
        if func not in (os.open,):
            warnings.warn(PytestWarning(f'(rm_rf) unknown function {func} when removing {path}:\n{type(exc)}: {exc}'))
        return False
    import stat

    def chmod_rw(p: str) -> None:
        mode = os.stat(p).st_mode
        os.chmod(p, mode | stat.S_IRUSR | stat.S_IWUSR)
    p = Path(path)
    if p.is_file():
        for parent in p.parents:
            chmod_rw(str(parent))
            if parent == start_path:
                break
    chmod_rw(str(path))
    func(path)
    return True
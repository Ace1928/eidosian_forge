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
def resolve_pkg_root_and_module_name(path: Path, *, consider_namespace_packages: bool=False) -> Tuple[Path, str]:
    """
    Return the path to the directory of the root package that contains the
    given Python file, and its module name:

        src/
            app/
                __init__.py
                core/
                    __init__.py
                    models.py

    Passing the full path to `models.py` will yield Path("src") and "app.core.models".

    If consider_namespace_packages is True, then we additionally check upwards in the hierarchy
    until we find a directory that is reachable from sys.path, which marks it as a namespace package:

    https://packaging.python.org/en/latest/guides/packaging-namespace-packages

    Raises CouldNotResolvePathError if the given path does not belong to a package (missing any __init__.py files).
    """
    pkg_path = resolve_package_path(path)
    if pkg_path is not None:
        pkg_root = pkg_path.parent
        if consider_namespace_packages:
            for parent in pkg_root.parents:
                if (parent / '__init__.py').is_file():
                    break
                if str(parent) in sys.path:
                    pkg_root = parent
                    break
        names = list(path.with_suffix('').relative_to(pkg_root).parts)
        if names[-1] == '__init__':
            names.pop()
        module_name = '.'.join(names)
        return (pkg_root, module_name)
    raise CouldNotResolvePathError(f'Could not resolve for {path}')
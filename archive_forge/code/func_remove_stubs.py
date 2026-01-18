import itertools
import os
from fnmatch import fnmatchcase
from glob import glob
from pathlib import Path
from typing import (
import _distutils_hack.override  # noqa: F401
from distutils import log
from distutils.util import convert_path
def remove_stubs(packages: List[str]) -> List[str]:
    """Remove type stubs (:pep:`561`) from a list of packages.

    >>> remove_stubs(["a", "a.b", "a-stubs", "a-stubs.b.c", "b", "c-stubs"])
    ['a', 'a.b', 'b']
    """
    return [pkg for pkg in packages if not pkg.split('.')[0].endswith('-stubs')]
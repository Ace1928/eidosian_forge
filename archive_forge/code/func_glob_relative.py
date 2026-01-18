import ast
import importlib
import os
import pathlib
import sys
from glob import iglob
from configparser import ConfigParser
from importlib.machinery import ModuleSpec
from itertools import chain
from typing import (
from pathlib import Path
from types import ModuleType
from distutils.errors import DistutilsOptionError
from .._path import same_path as _same_path
from ..warnings import SetuptoolsWarning
def glob_relative(patterns: Iterable[str], root_dir: Optional[_Path]=None) -> List[str]:
    """Expand the list of glob patterns, but preserving relative paths.

    :param list[str] patterns: List of glob patterns
    :param str root_dir: Path to which globs should be relative
                         (current directory by default)
    :rtype: list
    """
    glob_characters = {'*', '?', '[', ']', '{', '}'}
    expanded_values = []
    root_dir = root_dir or os.getcwd()
    for value in patterns:
        if any((char in value for char in glob_characters)):
            glob_path = os.path.abspath(os.path.join(root_dir, value))
            expanded_values.extend(sorted((os.path.relpath(path, root_dir).replace(os.sep, '/') for path in iglob(glob_path, recursive=True))))
        else:
            path = os.path.relpath(value, root_dir).replace(os.sep, '/')
            expanded_values.append(path)
    return expanded_values
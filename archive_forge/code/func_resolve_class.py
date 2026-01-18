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
def resolve_class(qualified_class_name: str, package_dir: Optional[Mapping[str, str]]=None, root_dir: Optional[_Path]=None) -> Callable:
    """Given a qualified class name, return the associated class object"""
    root_dir = root_dir or os.getcwd()
    idx = qualified_class_name.rfind('.')
    class_name = qualified_class_name[idx + 1:]
    pkg_name = qualified_class_name[:idx]
    _parent_path, path, module_name = _find_module(pkg_name, package_dir, root_dir)
    module = _load_spec(_find_spec(module_name, path), module_name)
    return getattr(module, class_name)
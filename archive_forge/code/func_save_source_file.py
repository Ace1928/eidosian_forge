import collections
import importlib.machinery
import io
import linecache
import pickletools
import platform
import types
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from enum import Enum
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import (
import torch
from torch.serialization import location_tag, normalize_storage_type
from torch.types import Storage
from torch.utils.hooks import RemovableHandle
from ._digraph import DiGraph
from ._importlib import _normalize_path
from ._mangling import demangle, is_mangled
from ._package_pickler import create_pickler
from ._stdlib import is_stdlib_module
from .find_file_dependencies import find_files_source_depends_on
from .glob_group import GlobGroup, GlobPattern
from .importer import Importer, OrderedImporter, sys_importer
from _mock import MockedObject
def save_source_file(self, module_name: str, file_or_directory: str, dependencies=True):
    """Adds the local file system ``file_or_directory`` to the source package to provide the code
        for ``module_name``.

        Args:
            module_name (str): e.g. ``"my_package.my_subpackage"``, code will be saved to provide code for this package.
            file_or_directory (str): the path to a file or directory of code. When a directory, all python files in the directory
                are recursively copied using :meth:`save_source_file`. If a file is named ``"/__init__.py"`` the code is treated
                as a package.
            dependencies (bool, optional): If ``True``, we scan the source for dependencies.
        """
    path = Path(file_or_directory)
    if path.is_dir():
        to_save = []
        module_path = module_name.replace('.', '/')
        for filename in path.glob('**/*.py'):
            relative_path = filename.relative_to(path).as_posix()
            archivename = module_path + '/' + relative_path
            submodule_name = None
            if filename.name == '__init__.py':
                submodule_name = archivename[:-len('/__init__.py')].replace('/', '.')
                is_package = True
            else:
                submodule_name = archivename[:-len('.py')].replace('/', '.')
                is_package = False
            to_save.append((submodule_name, _read_file(str(filename)), is_package, dependencies))
        for item in to_save:
            self.save_source_string(*item)
    else:
        is_package = path.name == '__init__.py'
        self.save_source_string(module_name, _read_file(file_or_directory), is_package, dependencies)
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
def get_rdeps(self, module_name: str) -> List[str]:
    """Return a list of all modules which depend on the module ``module_name``.

        Returns:
            A list containing the names of modules which depend on ``module_name``.
        """
    if module_name in self.dependency_graph._pred.keys():
        return list(self.dependency_graph._pred[module_name].keys())
    else:
        return []
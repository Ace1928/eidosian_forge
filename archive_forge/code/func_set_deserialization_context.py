import builtins
import importlib
import importlib.machinery
import inspect
import io
import linecache
import os.path
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO, Callable, cast, Dict, Iterable, List, Optional, Union
from weakref import WeakValueDictionary
import torch
from torch.serialization import _get_restore_location, _maybe_decode_ascii
from ._directory_reader import DirectoryReader
from ._importlib import (
from ._mangling import demangle, PackageMangler
from ._package_unpickler import PackageUnpickler
from .file_structure_representation import _create_directory_from_file_list, Directory
from .glob_group import GlobPattern
from .importer import Importer
@contextmanager
def set_deserialization_context():
    self.storage_context = storage_context
    self.last_map_location = map_location
    try:
        yield
    finally:
        self.storage_context = None
        self.last_map_location = None
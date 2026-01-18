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
def file_structure(self, *, include: 'GlobPattern'='**', exclude: 'GlobPattern'=()) -> Directory:
    """Returns a file structure representation of package's zipfile.

        Args:
            include (Union[List[str], str]): An optional string e.g. ``"my_package.my_subpackage"``, or optional list of strings
                for the names of the files to be included in the zipfile representation. This can also be
                a glob-style pattern, as described in :meth:`PackageExporter.mock`

            exclude (Union[List[str], str]): An optional pattern that excludes files whose name match the pattern.

        Returns:
            :class:`Directory`
        """
    return _create_directory_from_file_list(self.filename, self.zip_reader.get_all_records(), include, exclude)
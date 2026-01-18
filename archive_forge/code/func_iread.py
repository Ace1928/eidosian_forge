import io
import re
import functools
import inspect
import os
import sys
import numbers
import warnings
from pathlib import Path, PurePath
from typing import (
from ase.atoms import Atoms
from importlib import import_module
from ase.parallel import parallel_function, parallel_generator
def iread(filename: NameOrFile, index: Any=None, format: str=None, parallel: bool=True, do_not_split_by_at_sign: bool=False, **kwargs) -> Iterable[Atoms]:
    """Iterator for reading Atoms objects from file.

    Works as the `read` function, but yields one Atoms object at a time
    instead of all at once."""
    if isinstance(filename, PurePath):
        filename = str(filename)
    if isinstance(index, str):
        index = string2index(index)
    filename, index = parse_filename(filename, index, do_not_split_by_at_sign)
    if index is None or index == ':':
        index = slice(None, None, None)
    if not isinstance(index, (slice, str)):
        index = slice(index, index + 1 or None)
    format = format or filetype(filename, read=isinstance(filename, str))
    io = get_ioformat(format)
    for atoms in _iread(filename, index, format, io, parallel=parallel, **kwargs):
        yield atoms
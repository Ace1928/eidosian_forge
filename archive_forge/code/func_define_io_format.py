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
def define_io_format(name, desc, code, *, module=None, ext=None, glob=None, magic=None, encoding=None, magic_regex=None):
    if module is None:
        module = name.replace('-', '_')
        format2modulename[name] = module

    def normalize_patterns(strings):
        if strings is None:
            strings = []
        elif isinstance(strings, (str, bytes)):
            strings = [strings]
        else:
            strings = list(strings)
        return strings
    fmt = IOFormat(name, desc, code, module_name='ase.io.' + module, encoding=encoding)
    fmt.extensions = normalize_patterns(ext)
    fmt.globs = normalize_patterns(glob)
    fmt.magic = normalize_patterns(magic)
    if magic_regex is not None:
        fmt.magic_regex = magic_regex
    for ext in fmt.extensions:
        if ext in extension2format:
            raise ValueError('extension "{}" already registered'.format(ext))
        extension2format[ext] = fmt
    ioformats[name] = fmt
    return fmt
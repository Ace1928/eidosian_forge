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
def match_magic(self, data: bytes) -> bool:
    if self.magic_regex:
        assert not self.magic, 'Define only one of magic and magic_regex'
        match = re.match(self.magic_regex, data, re.M | re.S)
        return match is not None
    from fnmatch import fnmatchcase
    return any((fnmatchcase(data, magic + b'*') for magic in self.magic))
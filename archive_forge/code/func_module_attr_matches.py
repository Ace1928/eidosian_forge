import fnmatch
import importlib.machinery
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set, Generator, Sequence, Iterable, Union
from .line import (
def module_attr_matches(self, name: str) -> Set[str]:
    """Only attributes which are modules to replace name with"""
    return self.attr_matches(name, only_modules=True)
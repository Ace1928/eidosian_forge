import fnmatch
import importlib.machinery
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set, Generator, Sequence, Iterable, Union
from .line import (
def find_coroutine(self) -> bool:
    if self.fully_loaded:
        return False
    try:
        next(self.find_iterator)
    except StopIteration:
        self.fully_loaded = True
    return True
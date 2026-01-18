from __future__ import annotations
import abc
import collections
import os
import typing as t
from ...util import (
from .. import (
def all_files(self, include_symlinked_directories: bool=False) -> list[str]:
    """Return a list of all file paths."""
    if include_symlinked_directories:
        return self.__paths
    return self.__files
from __future__ import annotations
import abc
import dataclasses
import enum
import os
import typing as t
from .constants import (
from .util import (
from .data import (
from .become import (
def get_python_path(self, version: str) -> str:
    """Return the path of the requested Python version."""
    return os.path.join(self.python_dir, f'python{version}')
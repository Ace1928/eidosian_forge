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
@property
def supported_pythons(self) -> list[str]:
    """Return a list of the supported Python versions."""
    versions = self.python.split(',') if self.python else []
    versions = [version for version in versions if version in SUPPORTED_PYTHON_VERSIONS]
    return versions
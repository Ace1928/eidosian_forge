from __future__ import annotations
import os
import pickle
import typing as t
from .constants import (
from .compat.packaging import (
from .compat.yaml import (
from .io import (
from .util import (
from .data import (
from .config import (
def parse_python_requires(value: t.Any) -> tuple[str, ...]:
    """Parse the given 'python_requires' version specifier and return the matching Python versions."""
    if not isinstance(value, str):
        raise ValueError('python_requires must must be of type `str` not type `%s`' % type(value))
    versions: tuple[str, ...]
    if value == 'default':
        versions = SUPPORTED_PYTHON_VERSIONS
    elif value == 'controller':
        versions = CONTROLLER_PYTHON_VERSIONS
    else:
        specifier_set = SpecifierSet(value)
        versions = tuple((version for version in SUPPORTED_PYTHON_VERSIONS if specifier_set.contains(Version(version))))
    return versions
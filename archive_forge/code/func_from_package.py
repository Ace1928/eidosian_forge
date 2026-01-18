import os
import pathlib
import tempfile
import functools
import contextlib
import types
import importlib
from typing import Union, Optional
from .abc import ResourceReader, Traversable
from ._adapters import wrap_spec
def from_package(package):
    """
    Return a Traversable object for the given package.

    """
    spec = wrap_spec(package)
    reader = spec.loader.get_resource_reader(spec.name)
    return reader.files()
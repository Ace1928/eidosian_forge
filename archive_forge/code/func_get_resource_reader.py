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
def get_resource_reader(package):
    """
    Return the package's loader if it's a ResourceReader.
    """
    spec = package.__spec__
    reader = getattr(spec.loader, 'get_resource_reader', None)
    if reader is None:
        return None
    return reader(spec.name)
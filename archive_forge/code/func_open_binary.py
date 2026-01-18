import functools
import os
import pathlib
import types
import warnings
from typing import Union, Iterable, ContextManager, BinaryIO, TextIO, Any
from . import _common
@deprecated
def open_binary(package: Package, resource: Resource) -> BinaryIO:
    """Return a file-like object opened for binary reading of the resource."""
    return (_common.files(package) / normalize_path(resource)).open('rb')
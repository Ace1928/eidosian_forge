import functools
import os
import pathlib
import types
import warnings
from typing import Union, Iterable, ContextManager, BinaryIO, TextIO, Any
from . import _common
@deprecated
def open_text(package: Package, resource: Resource, encoding: str='utf-8', errors: str='strict') -> TextIO:
    """Return a file-like object opened for text reading of the resource."""
    return (_common.files(package) / normalize_path(resource)).open('r', encoding=encoding, errors=errors)
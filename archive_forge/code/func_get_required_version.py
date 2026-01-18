from pkgutil import extend_path
import sys
import os
import importlib
import types
from . import _gi
from ._gi import _API  # noqa: F401
from ._gi import Repository
from ._gi import PyGIDeprecationWarning  # noqa: F401
from ._gi import PyGIWarning  # noqa: F401
def get_required_version(namespace):
    return _versions.get(namespace, None)
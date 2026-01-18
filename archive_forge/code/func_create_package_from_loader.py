import abc
import importlib
import io
import sys
import types
import pathlib
import contextlib
from . import data01
from ..abc import ResourceReader
from .compat.py39 import import_helper, os_helper
from . import zip as zip_
from importlib.machinery import ModuleSpec
def create_package_from_loader(loader, is_package=True):
    name = 'testingpackage'
    module = types.ModuleType(name)
    spec = ModuleSpec(name, loader, origin='does-not-exist', is_package=is_package)
    module.__spec__ = spec
    module.__loader__ = loader
    return module
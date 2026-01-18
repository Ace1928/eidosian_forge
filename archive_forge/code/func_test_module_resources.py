import textwrap
import unittest
import warnings
import importlib
import contextlib
import importlib_resources as resources
from ..abc import Traversable
from . import data01
from . import util
from . import _path
from .compat.py39 import os_helper
from .compat.py312 import import_helper
def test_module_resources(self):
    """
        A module can have resources found adjacent to the module.
        """
    spec = {'mod.py': '', 'res.txt': 'resources are the best'}
    _path.build(spec, self.site_dir)
    import mod
    actual = resources.files(mod).joinpath('res.txt').read_text(encoding='utf-8')
    assert actual == spec['res.txt']
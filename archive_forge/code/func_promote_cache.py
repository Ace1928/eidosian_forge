import importlib
import logging
import os
import sys
import types
from io import StringIO
from typing import Any, Dict, List
import breezy
from .. import osutils, plugin, tests
from . import test_bar
def promote_cache(self, directory):
    """Move bytecode files out of __pycache__ in given directory."""
    cache_dir = os.path.join(directory, '__pycache__')
    if os.path.isdir(cache_dir):
        for name in os.listdir(cache_dir):
            magicless_name = '.'.join(name.split('.')[0::name.count('.')])
            rel = osutils.relpath(self.test_dir, cache_dir)
            self.log('moving %s in %s to %s', name, rel, magicless_name)
            os.rename(os.path.join(cache_dir, name), os.path.join(directory, magicless_name))
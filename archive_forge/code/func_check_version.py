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
def check_version(self, expected, source=None, name='plugin'):
    self.setup_plugin(source)
    self.assertEqual(expected, plugins[name].__version__)
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
def test_final_fallback__version__with_version_info(self):
    self.setup_plugin("version_info = (1, 2, 3, 'final', 2)")
    plugin = breezy.plugin.plugins()['plugin']
    self.assertEqual('1.2.3.2', plugin.__version__)
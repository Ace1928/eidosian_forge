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
def test_no_version_info___version__(self):
    self.setup_plugin()
    plugin = breezy.plugin.plugins()['plugin']
    self.assertEqual('unknown', plugin.__version__)
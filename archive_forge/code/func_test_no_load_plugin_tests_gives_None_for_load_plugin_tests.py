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
def test_no_load_plugin_tests_gives_None_for_load_plugin_tests(self):
    self.setup_plugin()
    loader = tests.TestUtil.TestLoader()
    p = plugin.plugins()['plugin']
    self.assertEqual(None, p.load_plugin_tests(loader))
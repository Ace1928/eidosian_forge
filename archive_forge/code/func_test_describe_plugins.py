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
def test_describe_plugins(self):

    class DummyModule:
        __doc__ = 'Hi there'

    class DummyPlugin:
        __version__ = '0.1.0'
        module = DummyModule()
    self.plugin_warnings = {'bad': ['Failed to load (just testing)']}
    self.plugins = {'good': DummyPlugin()}
    self.assertEqual('bad (failed to load)\n  ** Failed to load (just testing)\n\ngood 0.1.0\n  Hi there\n\n', ''.join(plugin.describe_plugins(state=self)))
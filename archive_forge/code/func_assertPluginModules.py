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
def assertPluginModules(self, plugin_dict):
    self.assertEqual({k[len(self.module_prefix):]: sys.modules[k] for k in sys.modules if k.startswith(self.module_prefix)}, plugin_dict)
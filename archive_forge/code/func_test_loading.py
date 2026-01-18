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
def test_loading(self):
    self.overrideEnv('BRZ_PLUGINS_AT', 'test_foo@non-standard-dir')
    self.load_with_paths(['standard'])
    self.assertTestFooLoadedFrom('non-standard-dir')
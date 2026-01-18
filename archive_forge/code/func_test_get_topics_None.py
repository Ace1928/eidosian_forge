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
def test_get_topics_None(self):
    """Searching for None returns an empty list."""
    index = plugin.PluginsHelpIndex()
    self.assertEqual([], index.get_topics(None))
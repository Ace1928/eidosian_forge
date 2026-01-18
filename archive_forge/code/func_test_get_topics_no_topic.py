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
def test_get_topics_no_topic(self):
    """Searching for something that is not a plugin returns []."""
    index = plugin.PluginsHelpIndex()
    self.assertEqual([], index.get_topics('nothing by this name'))
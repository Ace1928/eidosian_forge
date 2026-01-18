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
def test_get_help_text_carriage_return(self):
    """ModuleHelpTopic.get_help_text adds a 
 if needed."""
    mod = FakeModule('two lines of help\nand more\n', 'demo')
    topic = plugin.ModuleHelpTopic(mod)
    self.assertEqual('two lines of help\nand more\n', topic.get_help_text())
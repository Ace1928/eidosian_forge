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
def test_get_help_topic(self):
    """The help topic for a plugin is its module name."""
    mod = FakeModule('two lines of help\nand more', 'breezy.plugins.demo')
    topic = plugin.ModuleHelpTopic(mod)
    self.assertEqual('demo', topic.get_help_topic())
    mod = FakeModule('two lines of help\nand more', 'breezy.plugins.foo_bar')
    topic = plugin.ModuleHelpTopic(mod)
    self.assertEqual('foo_bar', topic.get_help_topic())
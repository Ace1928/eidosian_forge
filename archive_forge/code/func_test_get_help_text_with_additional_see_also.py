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
def test_get_help_text_with_additional_see_also(self):
    mod = FakeModule('two lines of help\nand more', 'demo')
    topic = plugin.ModuleHelpTopic(mod)
    self.assertEqual('two lines of help\nand more\n\n:See also: bar, foo\n', topic.get_help_text(['foo', 'bar']))
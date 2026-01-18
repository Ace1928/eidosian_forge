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
def test_duplicates_are_removed(self):
    self.check_path([self.user, self.core, self.site], ['+user', '+user'])
    self.check_path([self.user, self.core, self.site], ['+user', '+user', '+core', '+user', '+site', '+site', '+core'])
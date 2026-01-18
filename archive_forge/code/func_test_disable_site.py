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
def test_disable_site(self):
    self.check_path([self.core], ['-site'])
    self.check_path([self.user, self.core], ['-site', '+user'])
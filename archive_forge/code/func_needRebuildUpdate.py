import linecache
import sys
import time
import types
from importlib import reload
from types import ModuleType
from typing import Dict
from twisted.python import log, reflect
def needRebuildUpdate(self):
    yn = self.lastRebuild < lastRebuild
    return yn
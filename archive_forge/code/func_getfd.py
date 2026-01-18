import os
import signal
import sys
import threading
import warnings
from . import spawn
from . import util
def getfd(self):
    self.ensure_running()
    return self._fd
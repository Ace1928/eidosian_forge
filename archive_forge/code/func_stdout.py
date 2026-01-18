import builtins
import errno
import io
import locale
import os
import time
import signal
import sys
import threading
import warnings
import contextlib
from time import monotonic as _time
import types
@stdout.setter
def stdout(self, value):
    self.output = value
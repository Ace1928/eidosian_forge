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
def check_returncode(self):
    """Raise CalledProcessError if the exit code is non-zero."""
    if self.returncode:
        raise CalledProcessError(self.returncode, self.args, self.stdout, self.stderr)
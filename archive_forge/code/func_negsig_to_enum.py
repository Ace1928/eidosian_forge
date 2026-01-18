import glob
import os
import signal
import sys
import time
from ._common import MACOS
from ._common import TimeoutExpired
from ._common import memoize
from ._common import sdiskusage
from ._common import usage_percent
from ._compat import PY3
from ._compat import ChildProcessError
from ._compat import FileNotFoundError
from ._compat import InterruptedError
from ._compat import PermissionError
from ._compat import ProcessLookupError
from ._compat import unicode
def negsig_to_enum(num):
    return num
import importlib
import importlib.util
import inspect
import json
import os
import platform
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from PIL import Image
import pytest
import matplotlib as mpl
from matplotlib import _c_internal_utils
from matplotlib.backend_tools import ToolToggleBase
from matplotlib.testing import subprocess_run_helper as _run_helper
def interrupter():
    if sys.platform == 'win32':
        import win32api
        win32api.GenerateConsoleCtrlEvent(0, 0)
    else:
        import signal
        os.kill(os.getpid(), signal.SIGINT)
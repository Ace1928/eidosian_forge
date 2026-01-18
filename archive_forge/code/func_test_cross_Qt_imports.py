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
def test_cross_Qt_imports():
    qt5_bindings = [dep for dep in ['PyQt5', 'PySide2'] if importlib.util.find_spec(dep) is not None]
    qt6_bindings = [dep for dep in ['PyQt6', 'PySide6'] if importlib.util.find_spec(dep) is not None]
    if len(qt5_bindings) == 0 or len(qt6_bindings) == 0:
        pytest.skip('need both QT6 and QT5 bindings')
    for qt5 in qt5_bindings:
        for qt6 in qt6_bindings:
            for pair in ([qt5, qt6], [qt6, qt5]):
                try:
                    _run_helper(_impl_test_cross_Qt_imports, *pair, timeout=_test_timeout)
                except subprocess.CalledProcessError as ex:
                    if ex.returncode == -signal.SIGSEGV:
                        continue
                    elif ex.returncode == -signal.SIGABRT:
                        continue
                    raise
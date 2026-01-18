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
@pytest.mark.skipif(sys.platform != 'linux', reason='this a linux-only test')
@pytest.mark.parametrize('env', _get_testable_interactive_backends())
def test_lazy_linux_headless(env):
    proc = _run_helper(_lazy_headless, env.pop('MPLBACKEND'), env.pop('BACKEND_DEPS'), timeout=_test_timeout, extra_env={**env, 'DISPLAY': '', 'WAYLAND_DISPLAY': ''})
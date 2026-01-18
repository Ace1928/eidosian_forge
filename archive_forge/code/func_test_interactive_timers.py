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
@pytest.mark.parametrize('env', _get_testable_interactive_backends())
def test_interactive_timers(env):
    if env['MPLBACKEND'] == 'gtk3cairo' and os.getenv('CI'):
        pytest.skip('gtk3cairo timers do not work in remote CI')
    if env['MPLBACKEND'] == 'wx':
        pytest.skip('wx backend is deprecated; tests failed on appveyor')
    _run_helper(_impl_test_interactive_timers, timeout=_test_timeout, extra_env=env)
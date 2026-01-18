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
@pytest.mark.skipif(sys.platform == 'win32', reason='appveyor tests fail; gh-22988 suggests reworking')
@pytest.mark.parametrize('env', _get_testable_interactive_backends())
@pytest.mark.parametrize('time_mem', [(0.0, 2000000), (0.1, 30000000)])
def test_figure_leak_20490(env, time_mem, request):
    pytest.importorskip('psutil', reason='psutil needed to run this test')
    pause_time, acceptable_memory_leakage = time_mem
    if env['MPLBACKEND'] == 'wx':
        pytest.skip('wx backend is deprecated; tests failed on appveyor')
    if env['MPLBACKEND'] == 'macosx':
        request.node.add_marker(pytest.mark.xfail(reason='macosx backend is leaky'))
    if env['MPLBACKEND'] == 'tkagg' and sys.platform == 'darwin':
        acceptable_memory_leakage += 11000000
    result = _run_helper(_test_figure_leak, str(pause_time), timeout=_test_timeout, extra_env=env)
    growth = int(result.stdout)
    assert growth <= acceptable_memory_leakage
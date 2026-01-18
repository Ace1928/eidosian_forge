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
@pytest.mark.parametrize('env', _blit_backends)
@pytest.mark.flaky(reruns=4)
def test_blitting_events(env):
    proc = _run_helper(_test_number_of_draws_script, timeout=_test_timeout, extra_env=env)
    ndraws = proc.stdout.count('DrawEvent')
    assert 0 < ndraws < 5
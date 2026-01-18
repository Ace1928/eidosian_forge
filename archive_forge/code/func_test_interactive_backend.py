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
@pytest.mark.parametrize('toolbar', ['toolbar2', 'toolmanager'])
@pytest.mark.flaky(reruns=3)
def test_interactive_backend(env, toolbar):
    if env['MPLBACKEND'] == 'macosx':
        if toolbar == 'toolmanager':
            pytest.skip('toolmanager is not implemented for macosx.')
    if env['MPLBACKEND'] == 'wx':
        pytest.skip('wx backend is deprecated; tests failed on appveyor')
    try:
        proc = _run_helper(_test_interactive_impl, json.dumps({'toolbar': toolbar}), timeout=_test_timeout, extra_env=env)
    except subprocess.CalledProcessError as err:
        pytest.fail('Subprocess failed to test intended behavior\n' + str(err.stderr))
    assert proc.stdout.count('CloseEvent') == 1
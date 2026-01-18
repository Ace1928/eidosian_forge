import os
import shutil
import subprocess
import sys
import pytest
import numpy as np
from numpy.testing import IS_WASM
@pytest.fixture(scope='module')
def install_temp(tmpdir_factory):
    if IS_WASM:
        pytest.skip('No subprocess')
    srcdir = os.path.join(os.path.dirname(__file__), 'examples', 'cython')
    build_dir = tmpdir_factory.mktemp('cython_test') / 'build'
    os.makedirs(build_dir, exist_ok=True)
    try:
        subprocess.check_call(['meson', '--version'])
    except FileNotFoundError:
        pytest.skip("No usable 'meson' found")
    if sys.platform == 'win32':
        subprocess.check_call(['meson', 'setup', '--buildtype=release', '--vsenv', str(srcdir)], cwd=build_dir)
    else:
        subprocess.check_call(['meson', 'setup', str(srcdir)], cwd=build_dir)
    subprocess.check_call(['meson', 'compile', '-vv'], cwd=build_dir)
    sys.path.append(str(build_dir))
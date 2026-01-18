import sys
import os
import pytest
from os.path import join as pathjoin, isfile, dirname
import subprocess
import numpy as np
from numpy.testing import assert_equal, IS_WASM
@pytest.mark.skipif(is_inplace, reason='Cannot test f2py command inplace')
@pytest.mark.xfail(reason='Test is unreliable')
@pytest.mark.parametrize('f2py_cmd', find_f2py_commands())
def test_f2py(f2py_cmd):
    stdout = subprocess.check_output([f2py_cmd, '-v'])
    assert_equal(stdout.strip(), np.__version__.encode('ascii'))
import sys
import sysconfig
import subprocess
import pkgutil
import types
import importlib
import warnings
import numpy as np
import numpy
import pytest
from numpy.testing import IS_WASM
@pytest.mark.skipif(IS_WASM, reason="can't start subprocess")
@pytest.mark.parametrize('name', ['testing'])
def test_import_lazy_import(name):
    """Make sure we can actually use the modules we lazy load.

    While not exported as part of the public API, it was accessible.  With the
    use of __getattr__ and __dir__, this isn't always true It can happen that
    an infinite recursion may happen.

    This is the only way I found that would force the failure to appear on the
    badly implemented code.

    We also test for the presence of the lazily imported modules in dir

    """
    exe = (sys.executable, '-c', 'import numpy; numpy.' + name)
    result = subprocess.check_output(exe)
    assert not result
    assert name in dir(np)
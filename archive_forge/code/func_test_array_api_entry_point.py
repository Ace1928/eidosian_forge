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
@pytest.mark.xfail(sysconfig.get_config_var('Py_DEBUG') not in (None, 0, '0'), reason='NumPy possibly built with `USE_DEBUG=True ./tools/travis-test.sh`, which does not expose the `array_api` entry point. See https://github.com/numpy/numpy/pull/19800')
def test_array_api_entry_point():
    """
    Entry point for Array API implementation can be found with importlib and
    returns the numpy.array_api namespace.
    """
    numpy_in_sitepackages = sysconfig.get_path('platlib') in np.__file__
    eps = importlib.metadata.entry_points()
    try:
        xp_eps = eps.select(group='array_api')
    except AttributeError:
        xp_eps = eps.get('array_api', [])
    if len(xp_eps) == 0:
        if numpy_in_sitepackages:
            msg = "No entry points for 'array_api' found"
            raise AssertionError(msg) from None
        return
    try:
        ep = next((ep for ep in xp_eps if ep.name == 'numpy'))
    except StopIteration:
        if numpy_in_sitepackages:
            msg = "'numpy' not in array_api entry points"
            raise AssertionError(msg) from None
        return
    xp = ep.load()
    msg = f"numpy entry point value '{ep.value}' does not point to our Array API implementation"
    assert xp is numpy.array_api, msg
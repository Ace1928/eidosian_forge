import asyncio
import gc
import os
import pytest
import numpy as np
import threading
import warnings
from numpy.testing import extbuild, assert_warns, IS_WASM
import sys
@pytest.mark.skipif(sys.version_info >= (3, 12), reason='no numpy.distutils')
def test_context_locality(get_module):
    if sys.implementation.name == 'pypy' and sys.pypy_version_info[:3] < (7, 3, 6):
        pytest.skip('no context-locality support in PyPy < 7.3.6')
    asyncio.run(async_test_context_locality(get_module))
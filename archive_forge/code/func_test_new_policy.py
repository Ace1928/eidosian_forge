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
@pytest.mark.skip(reason='too slow, see gh-23975')
def test_new_policy(get_module):
    a = np.arange(10)
    orig_policy_name = np.core.multiarray.get_handler_name(a)
    orig_policy = get_module.set_secret_data_policy()
    b = np.arange(10)
    assert np.core.multiarray.get_handler_name(b) == 'secret_data_allocator'
    if orig_policy_name == 'default_allocator':
        assert np.core.test('full', verbose=1, extra_argv=[])
        assert np.ma.test('full', verbose=1, extra_argv=[])
    get_module.set_old_policy(orig_policy)
    c = np.arange(10)
    assert np.core.multiarray.get_handler_name(c) == orig_policy_name
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
def test_default_policy_singleton(get_module):
    get_handler_name = np.core.multiarray.get_handler_name
    orig_policy = get_module.set_old_policy(None)
    assert get_handler_name() == 'default_allocator'
    def_policy_1 = get_module.set_old_policy(None)
    assert get_handler_name() == 'default_allocator'
    def_policy_2 = get_module.set_old_policy(orig_policy)
    assert def_policy_1 is def_policy_2 is get_module.get_default_policy()
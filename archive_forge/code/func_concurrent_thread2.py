import asyncio
import gc
import os
import pytest
import numpy as np
import threading
import warnings
from numpy.testing import extbuild, assert_warns, IS_WASM
import sys
def concurrent_thread2(get_module, event):
    event.wait()
    assert np.core.multiarray.get_handler_name() == 'default_allocator'
    get_module.set_secret_data_policy()
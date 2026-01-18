import contextlib
import os
import signal
import subprocess
import sys
import weakref
import pyarrow as pa
import pytest
@pytest.mark.parametrize('pool_factory', supported_factories())
def test_debug_memory_pool_trap(pool_factory):
    res = run_debug_memory_pool(pool_factory.__name__, 'trap')
    if os.name == 'posix':
        assert res.returncode == -signal.SIGTRAP
    else:
        assert res.returncode != 0
    assert 'Wrong size on deallocation' in res.stderr
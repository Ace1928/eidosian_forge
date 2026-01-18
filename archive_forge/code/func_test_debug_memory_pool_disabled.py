import contextlib
import os
import signal
import subprocess
import sys
import weakref
import pyarrow as pa
import pytest
@pytest.mark.parametrize('pool_factory', supported_factories())
def test_debug_memory_pool_disabled(pool_factory):
    res = run_debug_memory_pool(pool_factory.__name__, '')
    if os.name == 'posix':
        assert res.returncode <= 0
    else:
        res.check_returncode()
    assert res.stderr == ''
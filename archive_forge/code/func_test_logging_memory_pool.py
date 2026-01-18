import contextlib
import os
import signal
import subprocess
import sys
import weakref
import pyarrow as pa
import pytest
def test_logging_memory_pool(capfd):
    pool = pa.logging_memory_pool(pa.default_memory_pool())
    check_allocated_bytes(pool)
    out, err = capfd.readouterr()
    assert err == ''
    assert out.count('Allocate:') > 0
    assert out.count('Allocate:') == out.count('Free:')
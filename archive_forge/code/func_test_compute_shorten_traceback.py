from __future__ import annotations
import traceback
from contextlib import contextmanager
import pytest
import dask
from dask.utils import shorten_traceback
@pytest.mark.parametrize('scheduler', ['threads', 'processes', 'sync'])
def test_compute_shorten_traceback(scheduler):
    d = dask.delayed(f3)()
    TEST_NAME = 'test_compute_shorten_traceback'
    if scheduler == 'processes' and (not tblib):
        remote_stack = ['reraise']
    else:
        remote_stack = ['f3', 'f2', 'f1']
    expect = [TEST_NAME, 'compute', *remote_stack]
    with assert_tb_levels(expect):
        dask.compute(d(), scheduler=scheduler)
    expect = [TEST_NAME, 'compute', 'compute', *remote_stack]
    with assert_tb_levels(expect):
        d.compute(scheduler=scheduler)
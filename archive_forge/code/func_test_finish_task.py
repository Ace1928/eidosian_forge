from __future__ import annotations
import pytest
import dask
from dask.local import finish_task, get_sync, sortkey, start_state_from_dask
from dask.order import order
from dask.utils_test import GetFunctionTestMixin, add, inc
def test_finish_task():
    dsk = {'x': 1, 'y': 2, 'z': (inc, 'x'), 'w': (add, 'z', 'y')}
    sortkey = order(dsk).get
    state = start_state_from_dask(dsk)
    state['ready'].remove('z')
    state['running'] = {'z', 'other-task'}
    task = 'z'
    result = 2
    state['cache']['z'] = result
    finish_task(dsk, task, state, set(), sortkey)
    assert state == {'cache': {'y': 2, 'z': 2}, 'dependencies': {'w': {'y', 'z'}, 'x': set(), 'y': set(), 'z': {'x'}}, 'finished': {'z'}, 'released': {'x'}, 'running': {'other-task'}, 'dependents': {'w': set(), 'x': {'z'}, 'y': {'w'}, 'z': {'w'}}, 'ready': ['w'], 'waiting': {}, 'waiting_data': {'y': {'w'}, 'z': {'w'}}}
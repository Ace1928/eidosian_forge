import time
from concurrent.futures import ThreadPoolExecutor
from panel.io.state import state
def test_as_cached_key_and_kwarg():

    def test_fn(a, i=[0]):
        i[0] += 1
        return i[0]
    assert state.as_cached('test', test_fn, a=1) == 1
    assert state.as_cached('test', test_fn, a=1) == 1
    assert state.as_cached('test', test_fn, a=2) == 2
    assert state.as_cached('test', test_fn, a=1) == 1
    assert state.as_cached('test', test_fn, a=2) == 2
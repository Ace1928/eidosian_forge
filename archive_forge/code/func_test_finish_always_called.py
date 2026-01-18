from __future__ import annotations
from dask.callbacks import Callback
from dask.local import get_sync
from dask.threaded import get as get_threaded
from dask.utils_test import add
def test_finish_always_called():
    flag = [False]

    class MyCallback(Callback):

        def _finish(self, dsk, state, errored):
            flag[0] = True
            assert errored
    dsk = {'x': (lambda: 1 / 0,)}
    try:
        with MyCallback():
            get_sync(dsk, 'x')
    except Exception as e:
        assert isinstance(e, ZeroDivisionError)
    assert flag[0]
    flag[0] = False
    try:
        with MyCallback():
            get_threaded(dsk, 'x')
    except Exception as e:
        assert isinstance(e, ZeroDivisionError)
    assert flag[0]

    def raise_keyboard():
        raise KeyboardInterrupt()
    dsk = {'x': (raise_keyboard,)}
    flag[0] = False
    try:
        with MyCallback():
            get_sync(dsk, 'x')
    except BaseException as e:
        assert isinstance(e, KeyboardInterrupt)
    assert flag[0]
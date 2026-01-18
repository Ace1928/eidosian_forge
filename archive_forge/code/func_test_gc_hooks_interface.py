import pytest
import types
import sys
import collections.abc
from functools import wraps
import gc
from .conftest import mock_sleep
from .. import (
from .. import _impl
def test_gc_hooks_interface(local_asyncgen_hooks):

    def one(agen):
        pass

    def two(agen):
        pass
    set_asyncgen_hooks(None, None)
    assert get_asyncgen_hooks() == (None, None)
    set_asyncgen_hooks(finalizer=two)
    assert get_asyncgen_hooks() == (None, two)
    set_asyncgen_hooks(firstiter=one)
    assert get_asyncgen_hooks() == (one, two)
    set_asyncgen_hooks(finalizer=None, firstiter=two)
    assert get_asyncgen_hooks() == (two, None)
    set_asyncgen_hooks(None, one)
    assert get_asyncgen_hooks() == (None, one)
    tup = (one, two)
    set_asyncgen_hooks(*tup)
    assert get_asyncgen_hooks() == tup
    with pytest.raises(TypeError):
        set_asyncgen_hooks(firstiter=42)
    with pytest.raises(TypeError):
        set_asyncgen_hooks(finalizer=False)

    def in_thread(results):
        results.append(get_asyncgen_hooks())
        set_asyncgen_hooks(two, one)
        results.append(get_asyncgen_hooks())
    from threading import Thread
    results = []
    thread = Thread(target=in_thread, args=(results,))
    thread.start()
    thread.join()
    assert results == [(None, None), (two, one)]
    assert get_asyncgen_hooks() == (one, two)
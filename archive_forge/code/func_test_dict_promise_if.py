from time import sleep
from pytest import raises, fixture
from threading import Event
from promise import (
from concurrent.futures import Future
from threading import Thread
from .utils import assert_exception
def test_dict_promise_if(promise_for_dict):
    p1 = Promise()
    p2 = Promise()
    d = {'a': p1, 'b': p2}
    pd = promise_for_dict(d)
    assert p1.is_pending
    assert p2.is_pending
    assert pd.is_pending
    p1.do_resolve(5)
    p1._wait()
    assert p1.is_fulfilled
    assert p2.is_pending
    assert pd.is_pending
    p2.do_resolve(10)
    p2._wait()
    assert p1.is_fulfilled
    assert p2.is_fulfilled
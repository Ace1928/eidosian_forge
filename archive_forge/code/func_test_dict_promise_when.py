from time import sleep
from pytest import raises, fixture
from threading import Event
from promise import (
from concurrent.futures import Future
from threading import Thread
from .utils import assert_exception
def test_dict_promise_when(promise_for_dict):
    p1 = Promise()
    p2 = Promise()
    d = {'a': p1, 'b': p2}
    pd1 = promise_for_dict(d)
    pd2 = promise_for_dict({'a': p1})
    pd3 = promise_for_dict({})
    assert p1.is_pending
    assert p2.is_pending
    assert pd1.is_pending
    assert pd2.is_pending
    pd3._wait()
    assert pd3.is_fulfilled
    p1.do_resolve(5)
    p1._wait()
    pd2._wait()
    assert p1.is_fulfilled
    assert p2.is_pending
    assert pd1.is_pending
    assert pd2.is_fulfilled
    p2.do_resolve(10)
    p2._wait()
    pd1._wait()
    assert p1.is_fulfilled
    assert p2.is_fulfilled
    assert pd1.is_fulfilled
    assert pd2.is_fulfilled
    assert 5 == p1.get()
    assert 10 == p2.get()
    assert 5 == pd1.get()['a']
    assert 5 == pd2.get()['a']
    assert 10 == pd1.get()['b']
    assert {} == pd3.get()
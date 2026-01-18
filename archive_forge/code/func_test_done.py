from time import sleep
from pytest import raises, fixture
from threading import Event
from promise import (
from concurrent.futures import Future
from threading import Thread
from .utils import assert_exception
def test_done():
    counter = [0]
    r = Promise()

    def inc(_):
        counter[0] += 1

    def dec(_):
        counter[0] -= 1

    def end(_):
        r.do_resolve(None)
    p = Promise()
    p.done(inc, dec)
    p.done(inc, dec)
    p.done(end)
    p.do_resolve(4)
    Promise.wait(r)
    assert counter[0] == 2
    r = Promise()
    counter = [0]
    p = Promise()
    p.done(inc, dec)
    p.done(inc, dec)
    p.done(None, end)
    p.do_reject(Exception())
    Promise.wait(r)
    assert counter[0] == -2
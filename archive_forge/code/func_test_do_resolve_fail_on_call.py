from time import sleep
from pytest import raises, fixture
from threading import Event
from promise import (
from concurrent.futures import Future
from threading import Thread
from .utils import assert_exception
def test_do_resolve_fail_on_call():

    def raises(resolve, reject):
        raise Exception('Fails')
    p1 = Promise(raises)
    assert not p1.is_fulfilled
    assert str(p1.reason) == 'Fails'
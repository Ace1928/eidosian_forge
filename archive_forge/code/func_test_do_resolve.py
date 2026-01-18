from time import sleep
from pytest import raises, fixture
from threading import Event
from promise import (
from concurrent.futures import Future
from threading import Thread
from .utils import assert_exception
def test_do_resolve():
    p1 = Promise(lambda resolve, reject: resolve(0))
    assert p1.get() == 0
    assert p1.is_fulfilled
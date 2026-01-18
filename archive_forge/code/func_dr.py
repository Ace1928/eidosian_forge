from time import sleep
from pytest import raises, fixture
from threading import Event
from promise import (
from concurrent.futures import Future
from threading import Thread
from .utils import assert_exception
def dr(reason, dtime):
    p = Promise()
    t = DelayedRejection(dtime, p, reason)
    t.start()
    return p
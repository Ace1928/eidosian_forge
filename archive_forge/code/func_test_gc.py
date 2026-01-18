import copy
import gc
import os
import sys
import time
from queue import Queue
from threading import Event, Thread
from unittest import mock
import pytest
from pytest import mark
import zmq
from zmq.tests import PYPY, BaseZMQTestCase, GreenTest, SkipTest
def test_gc(self):
    """test close&term by garbage collection alone"""
    if PYPY:
        raise SkipTest("GC doesn't work ")

    def gcf():

        def inner():
            ctx = self.Context()
            ctx.socket(zmq.PUSH)
        inner()
        gc.collect()
    t = Thread(target=gcf)
    t.start()
    t.join(timeout=1)
    assert not t.is_alive(), 'Garbage collection should have cleaned up context'
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
def test_instance_threadsafe(self):
    self.context.term()
    q = Queue()

    class SlowContext(self.Context):

        def __init__(self, *a, **kw):
            time.sleep(1)
            super().__init__(*a, **kw)

    def f():
        q.put(SlowContext.instance())
    N = 16
    threads = [Thread(target=f) for i in range(N)]
    [t.start() for t in threads]
    ctx = SlowContext.instance()
    assert isinstance(ctx, SlowContext)
    for i in range(N):
        thread_ctx = q.get(timeout=5)
        assert thread_ctx is ctx
    ctx.term()
    [t.join(timeout=5) for t in threads]
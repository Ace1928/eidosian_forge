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
@mark.skipif(sys.platform.startswith('win'), reason='Segfaults on Windows')
def test_destroy(self):
    """Context.destroy should close sockets"""
    ctx = self.Context()
    sockets = [ctx.socket(zmq.REP) for i in range(65)]
    [s.close() for s in sockets[::2]]
    ctx.destroy()
    time.sleep(0.01)
    for s in sockets:
        assert s.closed
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
def test_many_sockets(self):
    """opening and closing many sockets shouldn't cause problems"""
    ctx = self.Context()
    for i in range(16):
        sockets = [ctx.socket(zmq.REP) for i in range(65)]
        [s.close() for s in sockets]
        time.sleep(0.01)
    ctx.term()
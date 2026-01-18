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
def test_destroy_no_sockets(self):
    ctx = self.Context()
    s = ctx.socket(zmq.PUB)
    s.bind_to_random_port('tcp://127.0.0.1')
    s.close()
    ctx.destroy()
    assert s.closed
    assert ctx.closed
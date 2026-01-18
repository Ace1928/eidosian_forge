import copy
import gc
import sys
import time
import zmq
from zmq.tests import PYPY, BaseZMQTestCase, SkipTest, skip_pypy
def test_multisend(self):
    """ensure that a message remains intact after multiple sends"""
    a, b = self.create_bound_pair(zmq.PAIR, zmq.PAIR)
    s = b'message'
    m = zmq.Frame(s)
    assert s == m.bytes
    a.send(m, copy=False)
    time.sleep(0.1)
    assert s == m.bytes
    a.send(m, copy=False)
    time.sleep(0.1)
    assert s == m.bytes
    a.send(m, copy=True)
    time.sleep(0.1)
    assert s == m.bytes
    a.send(m, copy=True)
    time.sleep(0.1)
    assert s == m.bytes
    for i in range(4):
        r = b.recv()
        assert s == r
    assert s == m.bytes
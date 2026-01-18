import threading
import time
import zmq
from zmq import devices
from zmq.tests import PYPY, BaseZMQTestCase
def test_monitor_subscribe(self):
    alice, bob, mon = self.build_device(b'out')
    alices = b'hello bob'.split()
    alice.send_multipart(alices)
    alices2 = b'hello again'.split()
    alice.send_multipart(alices2)
    alices3 = b'hello again and again'.split()
    alice.send_multipart(alices3)
    bobs = self.recv_multipart(bob)
    assert alices == bobs
    bobs = self.recv_multipart(bob)
    assert alices2 == bobs
    bobs = self.recv_multipart(bob)
    assert alices3 == bobs
    bobs = b'hello alice'.split()
    bob.send_multipart(bobs)
    alices = self.recv_multipart(alice)
    assert alices == bobs
    mons = self.recv_multipart(mon)
    assert [b'out'] + bobs == mons
    self.teardown_device()
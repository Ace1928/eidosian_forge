import contextlib
import os
import time
from threading import Thread
import zmq
from zmq.tests import PYPY, BaseZMQTestCase, SkipTest
from zmq.utils import z85
def test_keypair(self):
    """test curve_keypair"""
    try:
        public, secret = zmq.curve_keypair()
    except zmq.ZMQError:
        raise SkipTest('CURVE unsupported')
    assert type(secret) == bytes
    assert type(public) == bytes
    assert len(secret) == 40
    assert len(public) == 40
    bsecret, bpublic = (z85.decode(key) for key in (public, secret))
    assert type(bsecret) == bytes
    assert type(bpublic) == bytes
    assert len(bsecret) == 32
    assert len(bpublic) == 32
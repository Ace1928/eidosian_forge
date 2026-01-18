import contextlib
import os
import time
from threading import Thread
import zmq
from zmq.tests import PYPY, BaseZMQTestCase, SkipTest
from zmq.utils import z85
def test_curve_public(self):
    """test curve_public"""
    try:
        public, secret = zmq.curve_keypair()
    except zmq.ZMQError:
        raise SkipTest('CURVE unsupported')
    if zmq.zmq_version_info() < (4, 2):
        raise SkipTest('curve_public is new in libzmq 4.2')
    derived_public = zmq.curve_public(secret)
    assert type(derived_public) == bytes
    assert len(derived_public) == 40
    bpublic = z85.decode(derived_public)
    assert type(bpublic) == bytes
    assert len(bpublic) == 32
    assert derived_public == public
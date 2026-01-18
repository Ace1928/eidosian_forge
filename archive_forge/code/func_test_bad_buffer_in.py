import copy
import gc
import sys
import time
import zmq
from zmq.tests import PYPY, BaseZMQTestCase, SkipTest, skip_pypy
def test_bad_buffer_in(self):
    """test using a bad object"""
    self.assertRaises(TypeError, zmq.Frame, 5)
    self.assertRaises(TypeError, zmq.Frame, object())
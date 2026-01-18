import copy
import gc
import sys
import time
import zmq
from zmq.tests import PYPY, BaseZMQTestCase, SkipTest, skip_pypy
def test_no_tracker(self):
    m = zmq.Frame(b'asdf', track=False)
    assert m.tracker is None
    m2 = copy.copy(m)
    assert m2.tracker is None
    self.assertRaises(ValueError, zmq.MessageTracker, m)
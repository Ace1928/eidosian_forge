import copy
import gc
import sys
import time
import zmq
from zmq.tests import PYPY, BaseZMQTestCase, SkipTest, skip_pypy
@skip_pypy
def test_above_30(self):
    """Message above 30 bytes are never copied by 0MQ."""
    for i in range(5, 16):
        s = 2 ** i * x
        rc = grc(s)
        m = zmq.Frame(s, copy=False)
        assert grc(s) == rc + 2
        del m
        await_gc(s, rc)
        assert grc(s) == rc
        del s
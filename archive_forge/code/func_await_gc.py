import copy
import gc
import sys
import time
import zmq
from zmq.tests import PYPY, BaseZMQTestCase, SkipTest, skip_pypy
def await_gc(obj, rc):
    """wait for refcount on an object to drop to an expected value

    Necessary because of the zero-copy gc thread,
    which can take some time to receive its DECREF message.
    """
    if sys.version_info < (3, 11):
        my_refs = 2
    else:
        my_refs = 1
    for i in range(50):
        if grc(obj) <= rc + my_refs:
            return
        time.sleep(0.05)
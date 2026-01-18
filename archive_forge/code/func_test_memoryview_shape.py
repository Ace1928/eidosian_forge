import copy
import gc
import sys
import time
import zmq
from zmq.tests import PYPY, BaseZMQTestCase, SkipTest, skip_pypy
def test_memoryview_shape(self):
    """memoryview shape info"""
    data = '§§¶•ªº˜µ¬˚…∆˙åß∂©œ∑´†≈ç√'.encode()
    n = len(data)
    f = zmq.Frame(data)
    view1 = f.buffer
    assert view1.ndim == 1
    assert view1.shape == (n,)
    assert view1.tobytes() == data
    view2 = memoryview(f)
    assert view2.ndim == 1
    assert view2.shape == (n,)
    assert view2.tobytes() == data
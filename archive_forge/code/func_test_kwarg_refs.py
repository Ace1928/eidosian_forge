from __future__ import print_function, absolute_import, division
import sys
import gc
import time
import weakref
import threading
import greenlet
from . import TestCase
from .leakcheck import fails_leakcheck
from .leakcheck import ignores_leakcheck
from .leakcheck import RUNNING_ON_MANYLINUX
def test_kwarg_refs(self):
    kwargs = {}
    g = greenlet.greenlet(lambda **kwargs: greenlet.getcurrent().parent.switch(**kwargs))
    for _ in range(100):
        g.switch(**kwargs)
    self.assertEqual(sys.getrefcount(kwargs), 2)
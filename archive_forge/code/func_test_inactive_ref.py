import gc
import weakref
import greenlet
from . import TestCase
from .leakcheck import fails_leakcheck
def test_inactive_ref(self):

    class inactive_greenlet(greenlet.greenlet):

        def __init__(self):
            greenlet.greenlet.__init__(self, run=self.run)

        def run(self):
            pass
    o = inactive_greenlet()
    o = weakref.ref(o)
    gc.collect()
    self.assertIsNone(o())
    self.assertFalse(gc.garbage, gc.garbage)
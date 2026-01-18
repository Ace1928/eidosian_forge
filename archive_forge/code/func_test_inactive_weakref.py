import gc
import weakref
import greenlet
from . import TestCase
def test_inactive_weakref(self):
    o = weakref.ref(greenlet.greenlet())
    gc.collect()
    self.assertEqual(o(), None)
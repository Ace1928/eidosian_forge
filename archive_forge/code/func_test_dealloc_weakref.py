import gc
import weakref
import greenlet
from . import TestCase
def test_dealloc_weakref(self):
    seen = []

    def worker():
        try:
            greenlet.getcurrent().parent.switch()
        finally:
            seen.append(g())
    g = greenlet.greenlet(worker)
    g.switch()
    g2 = greenlet.greenlet(lambda: None, g)
    g = weakref.ref(g2)
    g2 = None
    self.assertEqual(seen, [None])
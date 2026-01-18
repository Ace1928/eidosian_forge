import gc
import weakref
import greenlet
from . import TestCase
from .leakcheck import fails_leakcheck
@fails_leakcheck
def test_finalizer_crash(self):

    class object_with_finalizer(object):

        def __del__(self):
            pass
    array = []
    parent = greenlet.getcurrent()

    def greenlet_body():
        greenlet.getcurrent().object = object_with_finalizer()
        try:
            parent.switch()
        except greenlet.GreenletExit:
            print('Got greenlet exit!')
        finally:
            del greenlet.getcurrent().object
    g = greenlet.greenlet(greenlet_body)
    g.array = array
    array.append(g)
    g.switch()
    del array
    del g
    greenlet.getcurrent()
    gc.collect()
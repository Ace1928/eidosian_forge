import gc
import weakref
import greenlet
from . import TestCase
from .leakcheck import fails_leakcheck
def greenlet_body():
    greenlet.getcurrent().object = object_with_finalizer()
    try:
        parent.switch()
    except greenlet.GreenletExit:
        print('Got greenlet exit!')
    finally:
        del greenlet.getcurrent().object
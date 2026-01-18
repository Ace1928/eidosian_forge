from collections import OrderedDict
from twisted.trial import unittest
from twisted.words.xish import utility
from twisted.words.xish.domish import Element
from twisted.words.xish.utility import EventDispatcher
def test_addOnetimeObserverInDispatch(self):
    """
        Test for registration of a onetime observer during dispatch.
        """
    d = EventDispatcher()
    msg = Element(('ns', 'message'))
    cb = CallbackTracker()

    def onMessage(msg):
        d.addOnetimeObserver('/message', cb.call)
    d.addOnetimeObserver('/message', onMessage)
    d.dispatch(msg)
    self.assertEqual(cb.called, 0)
    d.dispatch(msg)
    self.assertEqual(cb.called, 1)
    d.dispatch(msg)
    self.assertEqual(cb.called, 1)
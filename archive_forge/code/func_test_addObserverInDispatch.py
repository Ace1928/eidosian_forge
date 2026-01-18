from collections import OrderedDict
from twisted.trial import unittest
from twisted.words.xish import utility
from twisted.words.xish.domish import Element
from twisted.words.xish.utility import EventDispatcher
def test_addObserverInDispatch(self):
    """
        Test for registration of an observer during dispatch.
        """
    d = EventDispatcher()
    msg = Element(('ns', 'message'))
    cb = CallbackTracker()

    def onMessage(_):
        d.addObserver('/message', cb.call)
    d.addOnetimeObserver('/message', onMessage)
    d.dispatch(msg)
    self.assertEqual(cb.called, 0)
    d.dispatch(msg)
    self.assertEqual(cb.called, 1)
    d.dispatch(msg)
    self.assertEqual(cb.called, 2)
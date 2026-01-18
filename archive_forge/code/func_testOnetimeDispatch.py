from collections import OrderedDict
from twisted.trial import unittest
from twisted.words.xish import utility
from twisted.words.xish.domish import Element
from twisted.words.xish.utility import EventDispatcher
def testOnetimeDispatch(self):
    d = EventDispatcher()
    msg = Element(('ns', 'message'))
    cb = CallbackTracker()
    d.addOnetimeObserver('/message', cb.call)
    d.dispatch(msg)
    self.assertEqual(cb.called, 1)
    d.dispatch(msg)
    self.assertEqual(cb.called, 1)
from collections import OrderedDict
from twisted.trial import unittest
from twisted.words.xish import utility
from twisted.words.xish.domish import Element
from twisted.words.xish.utility import EventDispatcher
def testOrderedXPathDispatch(self):
    d = EventDispatcher()
    cb = OrderedCallbackTracker()
    d.addObserver('/message/body', cb.call2)
    d.addObserver('/message', cb.call3, -1)
    d.addObserver('/message/body', cb.call1, 1)
    msg = Element(('ns', 'message'))
    msg.addElement('body')
    d.dispatch(msg)
    self.assertEqual(cb.callList, [cb.call1, cb.call2, cb.call3], 'Calls out of order: %s' % repr([c.__name__ for c in cb.callList]))
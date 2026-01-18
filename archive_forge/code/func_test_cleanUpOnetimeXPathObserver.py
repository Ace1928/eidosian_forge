from collections import OrderedDict
from twisted.trial import unittest
from twisted.words.xish import utility
from twisted.words.xish.domish import Element
from twisted.words.xish.utility import EventDispatcher
def test_cleanUpOnetimeXPathObserver(self):
    """
        Test observer clean-up after onetime XPath events.
        """
    d = EventDispatcher()
    cb = CallbackTracker()
    msg = Element((None, 'message'))
    d.addOnetimeObserver('/message', cb.call)
    d.dispatch(msg)
    self.assertEqual(1, cb.called)
    self.assertEqual(0, len(d._xpathObservers.pop(0)))
from io import StringIO
from twisted.internet import defer
from twisted.python import log
from twisted.python.reflect import qual
from twisted.spread import flavors, jelly, pb
from twisted.test.iosim import connectedServerAndClient
from twisted.trial import unittest
def test_jellyFailure(self):
    """
        Test that an exception which is a subclass of L{pb.Error} has more
        information passed across the network to the calling side.
        """

    def failureJelly(fail):
        fail.trap(JellyError)
        self.assertNotIsInstance(fail.type, str)
        self.assertIsInstance(fail.value, fail.type)
        return 43
    return self._testImpl('jelly', 43, failureJelly)
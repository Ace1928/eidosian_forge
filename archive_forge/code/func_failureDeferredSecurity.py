from io import StringIO
from twisted.internet import defer
from twisted.python import log
from twisted.python.reflect import qual
from twisted.spread import flavors, jelly, pb
from twisted.test.iosim import connectedServerAndClient
from twisted.trial import unittest
def failureDeferredSecurity(fail):
    fail.trap(SecurityError)
    self.assertNotIsInstance(fail.type, str)
    self.assertIsInstance(fail.value, fail.type)
    return 43000
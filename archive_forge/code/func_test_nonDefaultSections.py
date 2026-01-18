import struct
from io import BytesIO
from zope.interface.verify import verifyClass
from twisted.internet import address, task
from twisted.internet.error import CannotListenError, ConnectionDone
from twisted.names import dns
from twisted.python.failure import Failure
from twisted.python.util import FancyEqMixin, FancyStrMixin
from twisted.test import proto_helpers
from twisted.test.testutils import ComparisonTestsMixin
from twisted.trial import unittest
def test_nonDefaultSections(self):
    """
        L{dns._compactRepr} displays sections which differ from their defaults.
        """
    m = self.messageFactory()
    m.section1 = [1, 1, 1]
    m.section2 = [2, 2, 2]
    self.assertEqual("<Foo alwaysShowField='AS' flags=flagTrue section1=[1, 1, 1] section2=[2, 2, 2]>", repr(m))
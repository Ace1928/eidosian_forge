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
def test_ednsMessageDecodeStripsOptRecords(self):
    """
        The L(_EDNSMessage} instance created by L{dns._EDNSMessage.decode} from
        an EDNS query never includes OPT records in the additional section.
        """
    m = self.messageFactory()
    m.fromStr(MessageEDNSQuery.bytes())
    self.assertEqual(m.additional, [])
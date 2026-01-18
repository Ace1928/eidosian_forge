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
def test_SSHFP(self):
    """
        The byte stream written by L{dns.Record_SSHFP.encode} can be used by
        L{dns.Record_SSHFP.decode} to reconstruct the state of the original
        L{dns.Record_SSHFP} instance.
        """
    fp = b'\xda9\xa3\xee^kK\r' + b'2U\xbf\xef\x95`\x18\x90\xaf\xd8\x07\t'
    rr = dns.Record_SSHFP(algorithm=dns.Record_SSHFP.ALGORITHM_DSS, fingerprintType=dns.Record_SSHFP.FINGERPRINT_TYPE_SHA1, fingerprint=fp)
    self._recordRoundtripTest(rr)
    self.assertEncodedFormat(b'\x02\x01' + fp, rr)
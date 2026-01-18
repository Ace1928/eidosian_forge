import copy
import operator
import socket
from functools import partial, reduce
from io import BytesIO
from struct import pack
from twisted.internet import defer, error, reactor
from twisted.internet.defer import succeed
from twisted.internet.testing import (
from twisted.names import authority, client, common, dns, server
from twisted.names.client import Resolver
from twisted.names.dns import SOA, Message, Query, Record_A, Record_SOA, RRHeader
from twisted.names.error import DomainError
from twisted.names.secondary import SecondaryAuthority, SecondaryAuthorityService
from twisted.python.compat import nativeString
from twisted.python.filepath import FilePath
from twisted.trial import unittest
def test_aRecords(self):
    """
        A records are loaded.
        """
    for dom, ip in [(b'example.com', '10.0.0.1'), (b'no-in.example.com', '10.0.0.2')]:
        [[rr], [], []] = self.successResultOf(self.auth.lookupAddress(dom))
        self.assertEqual(dns.Record_A(ip), rr.payload)
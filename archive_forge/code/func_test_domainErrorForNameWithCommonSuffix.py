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
def test_domainErrorForNameWithCommonSuffix(self):
    """
        L{FileAuthority} lookup methods errback with L{DomainError} if
        the requested C{name} shares a common suffix with its zone but
        is not actually a descendant of its zone, in terms of its
        sequence of DNS name labels. eg www.the-example.com has
        nothing to do with the zone example.com.
        """
    testDomain = test_domain_com
    testDomainName = b'nonexistent.prefix-' + testDomain.soa[0]
    f = self.failureResultOf(testDomain.lookupAddress(testDomainName))
    self.assertIsInstance(f.value, DomainError)
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
def test_mailExchangeRecord(self):
    """
        The DNS client can issue an MX query and receive a response including
        an MX record as well as any A record hints.
        """
    return self.namesTest(self.resolver.lookupMailExchange(b'test-domain.com'), [dns.Record_MX(10, b'host.test-domain.com', ttl=19283784), dns.Record_A(b'123.242.1.5', ttl=19283784), dns.Record_A(b'0.255.0.255', ttl=19283784)])
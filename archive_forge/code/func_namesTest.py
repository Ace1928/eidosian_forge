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
def namesTest(self, querying, expectedRecords):
    """
        Assert that the DNS response C{querying} will eventually fire with
        contains exactly a certain collection of records.

        @param querying: A L{Deferred} returned from one of the DNS client
            I{lookup} methods.

        @param expectedRecords: A L{list} of L{IRecord} providers which must be
            in the response or the test will be failed.

        @return: A L{Deferred} that fires when the assertion has been made.  It
            fires with a success result if the assertion succeeds and with a
            L{Failure} if it fails.
        """

    def checkResults(response):
        receivedRecords = justPayload(response)
        self.assertEqual(set(expectedRecords), set(receivedRecords))
    querying.addCallback(checkResults)
    return querying
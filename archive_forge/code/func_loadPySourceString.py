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
def loadPySourceString(self, s):
    """
        Create a new L{twisted.names.authority.PySourceAuthority} from C{s}.

        @param s: A string with BIND zone data in a Python source file.
        @type s: L{str}

        @return: a new bind authority
        @rtype: L{twisted.names.authority.PySourceAuthority}
        """
    fp = FilePath(self.mktemp())
    with open(fp.path, 'w') as f:
        f.write(s)
    return authority.PySourceAuthority(fp.path)
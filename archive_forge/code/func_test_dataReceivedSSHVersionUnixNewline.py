import binascii
import re
import string
import struct
import types
from hashlib import md5, sha1, sha256, sha384, sha512
from typing import Dict, List, Optional, Tuple, Type
from twisted import __version__ as twisted_version
from twisted.conch.error import ConchError
from twisted.conch.ssh import _kex, address, service
from twisted.internet import defer
from twisted.protocols import loopback
from twisted.python import randbytes
from twisted.python.compat import iterbytes
from twisted.python.randbytes import insecureRandom
from twisted.python.reflect import requireModule
from twisted.test import proto_helpers
from twisted.trial.unittest import TestCase
def test_dataReceivedSSHVersionUnixNewline(self):
    """
        It can parse the SSH version string even when it ends only in
        Unix newlines (CR) and does not follows the RFC 4253 to use
        network newlines (CR LF).
        """
    sut = MockTransportBase()
    sut.makeConnection(proto_helpers.StringTransport())
    sut.dataReceived(b'SSH-2.0-PoorSSHD Some-comment here\nmore-data')
    self.assertTrue(sut.gotVersion)
    self.assertEqual(sut.otherVersionString, b'SSH-2.0-PoorSSHD Some-comment here')
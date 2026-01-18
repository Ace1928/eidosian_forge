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
def test_noPayloadSERVICE_ACCEPT(self):
    """
        Some commercial SSH servers don't send a payload with the
        SERVICE_ACCEPT message.  Conch pretends that it got the correct
        name of the service.
        """
    self.proto.instance = MockService()
    self.proto.ssh_SERVICE_ACCEPT(b'')
    self.assertTrue(self.proto.instance.started)
    self.assertEqual(len(self.packets), 0)
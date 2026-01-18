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
def checkDisconnected(self, kind=None):
    """
        Helper function to check if the transport disconnected.
        """
    if kind is None:
        kind = transport.DISCONNECT_PROTOCOL_ERROR
    self.assertEqual(self.packets[-1][0], transport.MSG_DISCONNECT)
    self.assertEqual(self.packets[-1][1][3:4], bytes((kind,)))
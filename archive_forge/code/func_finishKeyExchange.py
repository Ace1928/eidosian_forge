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
def finishKeyExchange(self, proto):
    """
        Deliver enough additional messages to C{proto} so that the key exchange
        which is started in L{SSHTransportBase.connectionMade} completes and
        non-key exchange messages can be sent and received.
        """
    proto.dataReceived(b'SSH-2.0-BogoClient-1.2i\r\n')
    proto.dispatchMessage(transport.MSG_KEXINIT, self._A_KEXINIT_MESSAGE)
    proto._keySetup(b'foo', b'bar')
    proto._keyExchangeState = proto._KEY_EXCHANGE_NONE
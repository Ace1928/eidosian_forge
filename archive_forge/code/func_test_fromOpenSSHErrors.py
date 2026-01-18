import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_fromOpenSSHErrors(self):
    """
        Tests for invalid key types.
        """
    badKey = b'-----BEGIN FOO PRIVATE KEY-----\nMIGkAgEBBDAtAi7I8j73WCX20qUM5hhHwHuFzYWYYILs2Sh8UZ+awNkARZ/Fu2LU\nLLl5RtOQpbWgBwYFK4EEACKhZANiAATU17sA9P5FRwSknKcFsjjsk0+E3CeXPYX0\nTk/M0HK3PpWQWgrO8JdRHP9eFE9O/23P8BumwFt7F/AvPlCzVd35VfraFT0o4cCW\nG0RqpQ+np31aKmeJshkcYALEchnU+tQ=\n-----END EC PRIVATE KEY-----'
    self.assertRaises(keys.BadKeyError, keys.Key._fromString_PRIVATE_OPENSSH, badKey, None)
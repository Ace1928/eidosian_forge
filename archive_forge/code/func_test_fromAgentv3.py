import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_fromAgentv3(self):
    """
        Test that keys are correctly generated from Agent v3 strings.
        """
    self._testPrivateFromString(keydata.privateRSA_agentv3, 'RSA', keydata.RSAData)
    self._testPrivateFromString(keydata.privateDSA_agentv3, 'DSA', keydata.DSAData)
    self.assertRaises(keys.BadKeyError, keys.Key.fromString, b'\x00\x00\x00\x07ssh-foo' + b'\x00\x00\x00\x01\x01' * 5)
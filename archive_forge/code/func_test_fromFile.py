import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_fromFile(self):
    """
        Test that fromFile works correctly.
        """
    self.assertEqual(keys.Key.fromFile(self.keyFile), keys.Key.fromString(keydata.privateRSA_lsh))
    self.assertRaises(keys.BadKeyError, keys.Key.fromFile, self.keyFile, 'bad_type')
    self.assertRaises(keys.BadKeyError, keys.Key.fromFile, self.keyFile, passphrase='unencrypted')
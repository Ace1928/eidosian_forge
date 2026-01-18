import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_noRegeneration(self):
    """
        L{keys._getPersistentRSAKey} will not regenerate the key if the key
        already exists.
        """
    tempDir = FilePath(self.mktemp())
    keyFile = tempDir.child('mykey.pem')
    key = keys._getPersistentRSAKey(keyFile, keySize=1024)
    self.assertEqual(key.size(), 1024)
    self.assertTrue(keyFile.exists())
    keyContent = keyFile.getContent()
    key = keys._getPersistentRSAKey(keyFile, keySize=2048)
    self.assertEqual(key.size(), 1024)
    self.assertEqual(keyFile.getContent(), keyContent)
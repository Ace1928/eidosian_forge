import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
@skipWithoutEd25519
def test_blobEd25519(self):
    """
        Return the over-the-wire SSH format of the Ed25519 public key.
        """
    from cryptography.hazmat.primitives import serialization
    publicBytes = self.ed25519Obj.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    self.assertEqual(keys.Key(self.ed25519Obj).blob(), common.NS(b'ssh-ed25519') + common.NS(publicBytes))
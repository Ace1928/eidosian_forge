import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
@skipWithoutEd25519
def test_fromBlobEd25519(self):
    """
        A public Ed25519 key is correctly generated from a public key blob.
        """
    ed25519PublicData = {'a': keydata.Ed25519Data['a']}
    ed25519Blob = common.NS(b'ssh-ed25519') + common.NS(ed25519PublicData['a'])
    ed25519Key = keys.Key.fromString(ed25519Blob)
    self.assertTrue(ed25519Key.isPublic())
    self.assertEqual(ed25519PublicData, ed25519Key.data())
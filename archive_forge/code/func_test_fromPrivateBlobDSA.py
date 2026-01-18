import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_fromPrivateBlobDSA(self):
    """
        A private DSA key is correctly generated from a private key blob.
        """
    dsaBlob = common.NS(b'ssh-dss') + common.MP(keydata.DSAData['p']) + common.MP(keydata.DSAData['q']) + common.MP(keydata.DSAData['g']) + common.MP(keydata.DSAData['y']) + common.MP(keydata.DSAData['x'])
    dsaKey = keys.Key._fromString_PRIVATE_BLOB(dsaBlob)
    self.assertFalse(dsaKey.isPublic())
    self.assertEqual(keydata.DSAData, dsaKey.data())
    self.assertEqual(dsaKey, keys.Key._fromString_PRIVATE_BLOB(dsaKey.privateBlob()))
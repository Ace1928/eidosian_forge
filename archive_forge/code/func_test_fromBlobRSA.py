import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_fromBlobRSA(self):
    """
        A public RSA key is correctly generated from a public key blob.
        """
    rsaPublicData = {'n': keydata.RSAData['n'], 'e': keydata.RSAData['e']}
    rsaBlob = common.NS(b'ssh-rsa') + common.MP(rsaPublicData['e']) + common.MP(rsaPublicData['n'])
    rsaKey = keys.Key.fromString(rsaBlob)
    self.assertTrue(rsaKey.isPublic())
    self.assertEqual(rsaPublicData, rsaKey.data())
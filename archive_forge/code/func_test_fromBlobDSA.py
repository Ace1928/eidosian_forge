import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_fromBlobDSA(self):
    """
        A public DSA key is correctly generated from a public key blob.
        """
    dsaPublicData = {'p': keydata.DSAData['p'], 'q': keydata.DSAData['q'], 'g': keydata.DSAData['g'], 'y': keydata.DSAData['y']}
    dsaBlob = common.NS(b'ssh-dss') + common.MP(dsaPublicData['p']) + common.MP(dsaPublicData['q']) + common.MP(dsaPublicData['g']) + common.MP(dsaPublicData['y'])
    dsaKey = keys.Key.fromString(dsaBlob)
    self.assertTrue(dsaKey.isPublic())
    self.assertEqual(dsaPublicData, dsaKey.data())
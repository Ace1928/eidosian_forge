import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_fromBlobECDSA(self):
    """
        Key.fromString generates ECDSA keys from blobs.
        """
    from cryptography import utils
    ecPublicData = {'x': keydata.ECDatanistp256['x'], 'y': keydata.ECDatanistp256['y'], 'curve': keydata.ECDatanistp256['curve']}
    ecblob = common.NS(ecPublicData['curve']) + common.NS(ecPublicData['curve'][-8:]) + common.NS(b'\x04' + utils.int_to_bytes(ecPublicData['x'], 32) + utils.int_to_bytes(ecPublicData['y'], 32))
    eckey = keys.Key.fromString(ecblob)
    self.assertTrue(eckey.isPublic())
    self.assertEqual(ecPublicData, eckey.data())
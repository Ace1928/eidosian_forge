import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_blobEC(self):
    """
        Return the over-the-wire SSH format of the EC public key.
        """
    from cryptography import utils
    byteLength = (self.ecObj.curve.key_size + 7) // 8
    self.assertEqual(keys.Key(self.ecObj).blob(), common.NS(keydata.ECDatanistp256['curve']) + common.NS(keydata.ECDatanistp256['curve'][-8:]) + common.NS(b'\x04' + utils.int_to_bytes(self.ecObj.private_numbers().public_numbers.x, byteLength) + utils.int_to_bytes(self.ecObj.private_numbers().public_numbers.y, byteLength)))
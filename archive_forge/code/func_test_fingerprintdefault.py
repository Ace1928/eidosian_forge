import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_fingerprintdefault(self):
    """
        Test that the fingerprint method returns fingerprint in
        L{FingerprintFormats.MD5-HEX} format by default.
        """
    self.assertEqual(keys.Key(self.rsaObj).fingerprint(), '85:25:04:32:58:55:96:9f:57:ee:fb:a8:1a:ea:69:da')
    self.assertEqual(keys.Key(self.dsaObj).fingerprint(), '63:15:b3:0e:e6:4f:50:de:91:48:3d:01:6b:b3:13:c1')
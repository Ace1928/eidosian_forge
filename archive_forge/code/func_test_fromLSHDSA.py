import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_fromLSHDSA(self):
    """
        DSA public and private key can be generated from LSHs.
        """
    self._testPublicPrivateFromString(keydata.publicDSA_lsh, keydata.privateDSA_lsh, 'DSA', keydata.DSAData)
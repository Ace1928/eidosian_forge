import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_fromOpenSSH_v1_format(self):
    """
        OpenSSH 6.5 introduced a newer "openssh-key-v1" private key format
        (made the default in OpenSSH 7.8).  Loading keys in this format
        produces identical results to loading the same keys in the old
        PEM-based format.
        """
    for old, new in ((keydata.privateRSA_openssh, keydata.privateRSA_openssh_new), (keydata.privateDSA_openssh, keydata.privateDSA_openssh_new), (keydata.privateECDSA_openssh, keydata.privateECDSA_openssh_new), (keydata.privateECDSA_openssh384, keydata.privateECDSA_openssh384_new), (keydata.privateECDSA_openssh521, keydata.privateECDSA_openssh521_new)):
        self.assertEqual(keys.Key.fromString(new), keys.Key.fromString(old))
    self.assertEqual(keys.Key.fromString(keydata.privateRSA_openssh_encrypted_new, passphrase=b'encrypted'), keys.Key.fromString(keydata.privateRSA_openssh_encrypted, passphrase=b'encrypted'))
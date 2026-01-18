import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_fromStringNormalizesUnicodePassphrase(self):
    """
        L{keys.Key.fromString} applies Normalization Form KC to Unicode
        passphrases.
        """
    key = keys.Key(self.rsaObj)
    key_data = key.toString('openssh', passphrase='verschlüsselt'.encode())
    self.assertEqual(keys.Key.fromString(key_data, passphrase='verschlüsselt'), key)
    self.assertRaises(keys.PassphraseNormalizationError, keys.Key.fromString, key_data, passphrase='unassigned \uffff')
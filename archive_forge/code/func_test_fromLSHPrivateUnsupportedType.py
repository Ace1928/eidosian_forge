import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_fromLSHPrivateUnsupportedType(self):
    """
        C{BadKeyError} exception is raised when private key has an unknown
        type.
        """
    sexp = sexpy.pack([[b'private-key', [b'bad-key', [b'p', b'2']]]])
    self.assertRaises(keys.BadKeyError, keys.Key.fromString, sexp)
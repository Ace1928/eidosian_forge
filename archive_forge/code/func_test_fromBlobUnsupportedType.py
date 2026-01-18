import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_fromBlobUnsupportedType(self):
    """
        A C{BadKeyError} error is raised whey the blob has an unsupported
        key type.
        """
    badBlob = common.NS(b'ssh-bad')
    self.assertRaises(keys.BadKeyError, keys.Key.fromString, badBlob)
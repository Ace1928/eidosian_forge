import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_fromPrivateBlobUnsupportedType(self):
    """
        C{BadKeyError} is raised when loading a private blob with an
        unsupported type.
        """
    badBlob = common.NS(b'ssh-bad')
    self.assertRaises(keys.BadKeyError, keys.Key._fromString_PRIVATE_BLOB, badBlob)
import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_blobNoKey(self):
    """
        C{RuntimeError} is raised when the blob is requested for a Key
        which is not wrapping anything.
        """
    badKey = keys.Key(None)
    self.assertRaises(RuntimeError, badKey.blob)
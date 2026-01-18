import random
import struct
import zipfile
from hashlib import md5
from twisted.python import filepath, zipstream
from twisted.trial import unittest
def test_closedArchive(self):
    """
        A closed ChunkingZipFile should raise a L{RuntimeError} when
        .readfile() is invoked.
        """
    czf = zipstream.ChunkingZipFile(self.makeZipFile(['something']), 'r')
    czf.close()
    self.assertRaises(RuntimeError, czf.readfile, 'something')
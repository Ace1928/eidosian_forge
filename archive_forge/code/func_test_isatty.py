import random
import struct
import zipfile
from hashlib import md5
from twisted.python import filepath, zipstream
from twisted.trial import unittest
def test_isatty(self):
    """
        zip files should not be ttys, so isatty() should be false
        """
    with self.getFileEntry('') as fileEntry:
        self.assertFalse(fileEntry.isatty())
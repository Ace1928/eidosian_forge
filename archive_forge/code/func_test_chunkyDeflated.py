import random
import struct
import zipfile
from hashlib import md5
from twisted.python import filepath, zipstream
from twisted.trial import unittest
def test_chunkyDeflated(self):
    """
        unzipIterChunky should unzip the given number of bytes per iteration on
        a deflated archive.
        """
    self._unzipIterChunkyTest(zipfile.ZIP_DEFLATED, 972, 23, 27)
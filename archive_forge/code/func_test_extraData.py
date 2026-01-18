import random
import struct
import zipfile
from hashlib import md5
from twisted.python import filepath, zipstream
from twisted.trial import unittest
def test_extraData(self):
    """
        readfile() should skip over 'extra' data present in the zip metadata.
        """
    fn = self.mktemp()
    with zipfile.ZipFile(fn, 'w') as zf:
        zi = zipfile.ZipInfo('0')
        extra_data = b'hello, extra'
        zi.extra = struct.pack('<hh', 42, len(extra_data)) + extra_data
        zf.writestr(zi, b'the real data')
    with zipstream.ChunkingZipFile(fn) as czf, czf.readfile('0') as zfe:
        self.assertEqual(zfe.read(), b'the real data')
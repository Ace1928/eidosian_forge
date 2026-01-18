import random
import struct
import zipfile
from hashlib import md5
from twisted.python import filepath, zipstream
from twisted.trial import unittest
def test_invalidHeader(self):
    """
        A zipfile entry with the wrong magic number should raise BadZipFile for
        readfile(), but that should not affect other files in the archive.
        """
    fn = self.makeZipFile(['test contents', 'more contents'])
    with zipfile.ZipFile(fn, 'r') as zf:
        zeroOffset = zf.getinfo('0').header_offset
    with open(fn, 'r+b') as scribble:
        scribble.seek(zeroOffset, 0)
        scribble.write(b'0' * 4)
    with zipstream.ChunkingZipFile(fn) as czf:
        self.assertRaises(zipfile.BadZipFile, czf.readfile, '0')
        with czf.readfile('1') as zfe:
            self.assertEqual(zfe.read(), b'more contents')
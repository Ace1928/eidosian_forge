import random
import struct
import zipfile
from hashlib import md5
from twisted.python import filepath, zipstream
from twisted.trial import unittest
def getFileEntry(self, contents):
    """
        Return an appropriate zip file entry
        """
    filename = self.mktemp()
    with zipfile.ZipFile(filename, 'w', self.compression) as z:
        z.writestr('content', contents)
    z = zipstream.ChunkingZipFile(filename, 'r')
    return z.readfile('content')
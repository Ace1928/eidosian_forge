import shutil
from base64 import b64decode
from twisted.persisted import dirdbm
from twisted.python import rebuild
from twisted.python.filepath import FilePath
from twisted.trial import unittest
def test_modificationTime(self) -> None:
    import time
    self.dbm[b'k'] = b'v'
    self.assertTrue(abs(time.time() - self.dbm.getModificationTime(b'k')) <= 3)
    self.assertRaises(KeyError, self.dbm.getModificationTime, b'nokey')
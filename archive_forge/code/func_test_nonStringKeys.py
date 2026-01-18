import shutil
from base64 import b64decode
from twisted.persisted import dirdbm
from twisted.python import rebuild
from twisted.python.filepath import FilePath
from twisted.trial import unittest
def test_nonStringKeys(self) -> None:
    """
        L{dirdbm.DirDBM} operations only support string keys: other types
        should raise a L{TypeError}.
        """
    self.assertRaises(TypeError, self.dbm.__setitem__, 2, '3')
    try:
        self.assertRaises(TypeError, self.dbm.__setitem__, '2', 3)
    except unittest.FailTest:
        self.assertIsInstance(self.dbm, dirdbm.Shelf)
    self.assertRaises(TypeError, self.dbm.__getitem__, 2)
    self.assertRaises(TypeError, self.dbm.__delitem__, 2)
    self.assertRaises(TypeError, self.dbm.has_key, 2)
    self.assertRaises(TypeError, self.dbm.__contains__, 2)
    self.assertRaises(TypeError, self.dbm.getModificationTime, 2)
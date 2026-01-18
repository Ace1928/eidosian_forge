import os
import unittest
from boto.s3.keyfile import KeyFile
from tests.integration.s3.mock_storage_service import MockConnection
from tests.integration.s3.mock_storage_service import MockBucket
def testSeekEnd(self):
    self.assertEqual(self.keyfile.read(4), self.contents[:4])
    self.keyfile.seek(0, os.SEEK_END)
    self.assertEqual(self.keyfile.read(1), '')
    self.keyfile.seek(-1, os.SEEK_END)
    self.assertEqual(self.keyfile.tell(), 9)
    self.assertEqual(self.keyfile.read(1), '9')
    try:
        self.keyfile.seek(-100, os.SEEK_END)
    except IOError as e:
        self.assertEqual(str(e), 'Invalid argument')
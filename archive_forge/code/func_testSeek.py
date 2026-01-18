import os
import unittest
from boto.s3.keyfile import KeyFile
from tests.integration.s3.mock_storage_service import MockConnection
from tests.integration.s3.mock_storage_service import MockBucket
def testSeek(self):
    self.assertEqual(self.keyfile.read(4), self.contents[:4])
    self.keyfile.seek(0)
    self.assertEqual(self.keyfile.read(4), self.contents[:4])
    self.keyfile.seek(5)
    self.assertEqual(self.keyfile.read(5), self.contents[5:])
    try:
        self.keyfile.seek(-5)
    except IOError as e:
        self.assertEqual(str(e), 'Invalid argument')
    self.keyfile.read(10)
    self.assertEqual(self.keyfile.read(20), '')
    self.keyfile.seek(50)
    self.assertEqual(self.keyfile.tell(), 50)
    self.assertEqual(self.keyfile.read(1), '')
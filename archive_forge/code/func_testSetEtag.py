import os
import unittest
from boto.s3.keyfile import KeyFile
from tests.integration.s3.mock_storage_service import MockConnection
from tests.integration.s3.mock_storage_service import MockBucket
def testSetEtag(self):
    self.keyfile.key.data = b'test'
    self.keyfile.key.set_etag()
    self.assertEqual(self.keyfile.key.etag, '098f6bcd4621d373cade4e832627b4f6')
    self.keyfile.key.etag = None
    self.keyfile.key.data = 'test'
    self.keyfile.key.set_etag()
    self.assertEqual(self.keyfile.key.etag, '098f6bcd4621d373cade4e832627b4f6')
import sys
import errno
import hashlib
from io import BytesIO
from unittest import mock
from unittest.mock import Mock
from libcloud.test import MockHttp, BodyStream, unittest
from libcloud.utils.py3 import PY2, StringIO, b, httplib, assertRaisesRegex
from libcloud.storage.base import DEFAULT_CONTENT_TYPE, StorageDriver
from libcloud.common.exceptions import RateLimitReachedError
from libcloud.test.storage.base import BaseRangeDownloadMockHttp
def test__get_hash_function(self):
    self.driver1.hash_type = 'md5'
    func = self.driver1._get_hash_function()
    self.assertTrue(func)
    self.driver1.hash_type = 'sha1'
    func = self.driver1._get_hash_function()
    self.assertTrue(func)
    try:
        self.driver1.hash_type = 'invalid-hash-function'
        func = self.driver1._get_hash_function()
    except RuntimeError:
        pass
    else:
        self.fail('Invalid hash type but exception was not thrown')
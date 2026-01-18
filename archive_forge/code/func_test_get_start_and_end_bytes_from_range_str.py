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
def test_get_start_and_end_bytes_from_range_str(self):
    mock_http = BaseRangeDownloadMockHttp(None, None)
    body = '0123456789'
    range_str = 'bytes=1-'
    result = mock_http._get_start_and_end_bytes_from_range_str(range_str, body)
    self.assertEqual(result, (1, len(body)))
    range_str = 'bytes=1-5'
    result = mock_http._get_start_and_end_bytes_from_range_str(range_str, body)
    self.assertEqual(result, (1, 5))
    range_str = 'bytes=3-5'
    result = mock_http._get_start_and_end_bytes_from_range_str(range_str, body)
    self.assertEqual(result, (3, 5))
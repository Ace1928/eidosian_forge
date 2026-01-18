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
def succeed_on_second(*_, **__) -> mock.MagicMock:
    nonlocal count
    count += 1
    if count > 1:
        successful_response = mock.MagicMock()
        successful_response.status_code = 200
        return successful_response
    else:
        raise RateLimitReachedError()
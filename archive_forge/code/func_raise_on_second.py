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
def raise_on_second(*_, **__):
    nonlocal count
    count += 1
    if count > 1:
        raise SecondException()
    else:
        raise RateLimitReachedError()
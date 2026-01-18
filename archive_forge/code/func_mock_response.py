import sys
import unittest
from unittest import mock
from libcloud.test import LibcloudTestCase
from libcloud.common.base import Response, LazyObject
from libcloud.common.exceptions import BaseHTTPError, RateLimitReachedError
def mock_response(self, code, headers={}):
    m = mock.MagicMock()
    m.request = mock.Mock()
    m.headers = headers
    m.status_code = code
    m.text = None
    return m
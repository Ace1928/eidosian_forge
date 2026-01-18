import os
import random
import unittest
import requests
import requests_mock
from libcloud.http import LibcloudConnection
from libcloud.utils.py3 import PY2, httplib, parse_qs, urlparse, urlquote, parse_qsl
from libcloud.common.base import Response
def prepared_request(self, method, url, body=None, headers=None, raw=False, stream=False):
    headers = self._normalize_headers(headers=headers)
    r_status, r_body, r_headers, r_reason = self._get_request(method, url, body, headers)
    with requests_mock.mock() as m:
        m.register_uri(method, url, text=r_body, reason=r_reason, headers=r_headers, status_code=r_status)
        super().prepared_request(method=method, url=url, body=body, headers=headers, raw=raw, stream=stream)
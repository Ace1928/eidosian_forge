import copy
import io
import os
from unittest import mock
import fixtures
from oslo_utils import strutils
import requests
import testtools
def mockSessionResponse(headers, content=None, status_code=None, version=None, request_headers={}):
    raw = mock.Mock()
    raw.version = version
    request = mock.Mock()
    request.headers = request_headers
    response = mock.Mock(spec=requests.Response, headers=headers, content=content, status_code=status_code, raw=raw, reason='', encoding='UTF-8', request=request)
    response.text = content
    return response
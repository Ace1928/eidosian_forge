import copy
import io
import os
from unittest import mock
import fixtures
from oslo_utils import strutils
import requests
import testtools
def mockSession(headers, content=None, status_code=None, version=None):
    session = mock.Mock(spec=requests.Session, verify=False, cert=('test_cert', 'test_key'))
    session.get_endpoint = mock.Mock(return_value='https://test')
    response = mockSessionResponse(headers, content, status_code, version)
    session.request = mock.Mock(return_value=response)
    return session
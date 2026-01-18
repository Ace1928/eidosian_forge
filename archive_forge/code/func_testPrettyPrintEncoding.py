import base64
import datetime
import sys
import contextlib
import unittest
import six
from six.moves import http_client
from six.moves import urllib_parse
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.py import base_api
from apitools.base.py import encoding
from apitools.base.py import exceptions
from apitools.base.py import http_wrapper
def testPrettyPrintEncoding(self):
    method_config = base_api.ApiMethodInfo(request_type_name='MessageWithTime', query_params=['timestamp'])
    service = FakeService()
    request = MessageWithTime(timestamp=datetime.datetime(2014, 10, 7, 12, 53, 13))
    global_params = StandardQueryParameters()
    http_request = service.PrepareHttpRequest(method_config, request, global_params=global_params)
    self.assertFalse('prettyPrint' in http_request.url)
    self.assertFalse('pp' in http_request.url)
    global_params.prettyPrint = False
    global_params.pp = False
    http_request = service.PrepareHttpRequest(method_config, request, global_params=global_params)
    self.assertTrue('prettyPrint=0' in http_request.url)
    self.assertTrue('pp=0' in http_request.url)
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
def testJsonResponse(self):
    method_config = base_api.ApiMethodInfo(response_type_name='SimpleMessage')
    service = FakeService()
    http_response = http_wrapper.Response(info={'status': '200'}, content='{"field": "abc"}', request_url='http://www.google.com')
    response_message = SimpleMessage(field='abc')
    self.assertEqual(response_message, service.ProcessHttpResponse(method_config, http_response))
    with service.client.JsonResponseModel():
        self.assertEqual(http_response.content, service.ProcessHttpResponse(method_config, http_response))
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
def testHttpError(self):

    def fakeMakeRequest(*unused_args, **unused_kwargs):
        return http_wrapper.Response(info={'status': http_client.BAD_REQUEST}, content='{"field": "abc"}', request_url='http://www.google.com')
    method_config = base_api.ApiMethodInfo(request_type_name='SimpleMessage', response_type_name='SimpleMessage')
    client = self.__GetFakeClient()
    service = FakeService(client=client)
    request = SimpleMessage()
    with mock(base_api.http_wrapper, 'MakeRequest', fakeMakeRequest):
        with self.assertRaises(exceptions.HttpBadRequestError) as err:
            service._RunMethod(method_config, request)
    http_error = err.exception
    self.assertEquals('http://www.google.com', http_error.url)
    self.assertEquals('{"field": "abc"}', http_error.content)
    self.assertEquals(method_config, http_error.method_config)
    self.assertEquals(request, http_error.request)
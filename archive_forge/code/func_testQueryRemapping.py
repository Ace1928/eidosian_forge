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
def testQueryRemapping(self):
    method_config = base_api.ApiMethodInfo(request_type_name='MessageWithRemappings', query_params=['remapped_field', 'enum_field'])
    request = MessageWithRemappings(str_field='foo', enum_field=MessageWithRemappings.AnEnum.value_one)
    http_request = FakeService().PrepareHttpRequest(method_config, request)
    result_params = urllib_parse.parse_qs(urllib_parse.urlparse(http_request.url).query)
    expected_params = {'enum_field': 'ONE%2FTWO', 'remapped_field': 'foo'}
    self.assertTrue(expected_params, result_params)
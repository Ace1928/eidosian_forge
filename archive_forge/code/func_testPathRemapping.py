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
def testPathRemapping(self):
    method_config = base_api.ApiMethodInfo(relative_path='parameters/{remapped_field}/remap/{enum_field}', request_type_name='MessageWithRemappings', path_params=['remapped_field', 'enum_field'])
    request = MessageWithRemappings(str_field='gonna', enum_field=MessageWithRemappings.AnEnum.value_one)
    service = FakeService()
    expected_url = service.client.url + 'parameters/gonna/remap/ONE%2FTWO'
    http_request = service.PrepareHttpRequest(method_config, request)
    self.assertEqual(expected_url, http_request.url)
    method_config.relative_path = 'parameters/{+remapped_field}/remap/{+enum_field}'
    expected_url = service.client.url + 'parameters/gonna/remap/ONE/TWO'
    http_request = service.PrepareHttpRequest(method_config, request)
    self.assertEqual(expected_url, http_request.url)
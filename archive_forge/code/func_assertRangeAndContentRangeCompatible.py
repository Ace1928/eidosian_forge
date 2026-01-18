import string
import unittest
import httplib2
import json
import mock
import six
from six.moves import http_client
from apitools.base.py import base_api
from apitools.base.py import exceptions
from apitools.base.py import gzip
from apitools.base.py import http_wrapper
from apitools.base.py import transfer
def assertRangeAndContentRangeCompatible(self, request, response):
    request_prefix = 'bytes='
    self.assertIn('range', request.headers)
    self.assertTrue(request.headers['range'].startswith(request_prefix))
    request_range = request.headers['range'][len(request_prefix):]
    response_prefix = 'bytes '
    self.assertIn('content-range', response.info)
    response_header = response.info['content-range']
    self.assertTrue(response_header.startswith(response_prefix))
    response_range = response_header[len(response_prefix):].partition('/')[0]
    msg = 'Request range ({0}) not a prefix of response_range ({1})'.format(request_range, response_range)
    self.assertTrue(response_range.startswith(request_range), msg=msg)
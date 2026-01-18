import functools
import json
import logging
from unittest import mock
import uuid
import fixtures
import io
from keystoneauth1 import session
from keystoneauth1 import token_endpoint
from oslo_utils import encodeutils
import requests
from requests_mock.contrib import fixture
from urllib import parse
from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
import testtools
from testtools import matchers
import types
import glanceclient
from glanceclient.common import http
from glanceclient.tests import utils
@mock.patch('keystoneauth1.adapter.Adapter.request')
def test_http_duplicate_content_type_headers(self, mock_ksarq):
    """Test proper handling of Content-Type headers.

        encode_headers() must be called as late as possible before a
        request is sent. If this principle is violated, and if any
        changes are made to the headers between encode_headers() and the
        actual request (for instance a call to
        _set_common_request_kwargs()), and if you're trying to set a
        Content-Type that is not equal to application/octet-stream (the
        default), it is entirely possible that you'll end up with two
        Content-Type headers defined (yours plus
        application/octet-stream). The request will go out the door with
        only one of them chosen seemingly at random.

        This situation only occurs in python3. This test will never fail
        in python2.
        """
    path = '/v2/images/my-image'
    headers = {'Content-Type': 'application/openstack-images-v2.1-json-patch'}
    data = '[{"value": "qcow2", "path": "/disk_format", "op": "replace"}]'
    self.mock.patch(self.endpoint + path)
    sess_http_client = self._create_session_client()
    sess_http_client.patch(path, headers=headers, data=data)
    ksarqh = mock_ksarq.call_args[1]['headers']
    self.assertEqual(1, [encodeutils.safe_decode(key) for key in ksarqh.keys()].count('Content-Type'))
    self.assertEqual(b'application/openstack-images-v2.1-json-patch', ksarqh[b'Content-Type'])
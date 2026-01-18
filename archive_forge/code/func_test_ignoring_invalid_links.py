import json
import re
from unittest import mock
from testtools import matchers
from keystoneauth1 import adapter
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1 import http_basic
from keystoneauth1 import noauth
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
from keystoneauth1 import token_endpoint
def test_ignoring_invalid_links(self):
    version_list = [{'id': 'v3.0', 'links': [{'href': V3_URL, 'rel': 'self'}], 'media-types': V3_MEDIA_TYPES, 'status': 'stable', 'updated': UPDATED}, {'id': 'v3.1', 'media-types': V3_MEDIA_TYPES, 'status': 'stable', 'updated': UPDATED}, {'media-types': V3_MEDIA_TYPES, 'status': 'stable', 'updated': UPDATED, 'links': [{'href': V3_URL, 'rel': 'self'}]}]
    self.requests_mock.get(BASE_URL, json={'versions': version_list})
    disc = discover.Discover(self.session, BASE_URL)
    versions = disc.raw_version_data()
    self.assertEqual(3, len(versions))
    versions = disc.version_data()
    self.assertEqual(1, len(versions))
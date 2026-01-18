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
def test_glance_version_data(self):
    mock = self.requests_mock.get(BASE_URL, status_code=200, json=GLANCE_EXAMPLES)
    disc = discover.Discover(self.session, BASE_URL)
    raw_data = disc.raw_version_data()
    clean_data = disc.version_data()
    self.assertEqual(5, len(raw_data))
    for v in raw_data:
        if v['id'] in ('v2.2', 'v1.1'):
            self.assertEqual(v['status'], discover.Status.CURRENT)
        elif v['id'] in ('v2.1', 'v2.0', 'v1.0'):
            self.assertEqual(v['status'], discover.Status.SUPPORTED)
        else:
            self.fail('Invalid version found')
    v1_url = '%sv1/' % BASE_URL
    v2_url = '%sv2/' % BASE_URL
    self.assertEqual(clean_data, [{'collection': None, 'max_microversion': None, 'min_microversion': None, 'next_min_version': None, 'not_before': None, 'version': (1, 0), 'url': v1_url, 'status': discover.Status.SUPPORTED, 'raw_status': discover.Status.SUPPORTED}, {'collection': None, 'max_microversion': None, 'min_microversion': None, 'next_min_version': None, 'not_before': None, 'version': (1, 1), 'url': v1_url, 'status': discover.Status.CURRENT, 'raw_status': discover.Status.CURRENT}, {'collection': None, 'max_microversion': None, 'min_microversion': None, 'next_min_version': None, 'not_before': None, 'version': (2, 0), 'url': v2_url, 'status': discover.Status.SUPPORTED, 'raw_status': discover.Status.SUPPORTED}, {'collection': None, 'max_microversion': None, 'min_microversion': None, 'next_min_version': None, 'not_before': None, 'version': (2, 1), 'url': v2_url, 'status': discover.Status.SUPPORTED, 'raw_status': discover.Status.SUPPORTED}, {'collection': None, 'max_microversion': None, 'min_microversion': None, 'next_min_version': None, 'not_before': None, 'version': (2, 2), 'url': v2_url, 'status': discover.Status.CURRENT, 'raw_status': discover.Status.CURRENT}])
    for ver in (2, 2.1, 2.2):
        for version in (disc.data_for(ver), disc.versioned_data_for(min_version=ver, max_version=(2, discover.LATEST))):
            self.assertEqual((2, 2), version['version'])
            self.assertEqual(discover.Status.CURRENT, version['raw_status'])
            self.assertEqual(v2_url, version['url'])
            self.assertEqual(v2_url, disc.url_for(ver))
    for ver in (1, 1.1):
        for version in (disc.data_for(ver), disc.versioned_data_for(min_version=ver, max_version=(1, discover.LATEST))):
            self.assertEqual((1, 1), version['version'])
            self.assertEqual(discover.Status.CURRENT, version['raw_status'])
            self.assertEqual(v1_url, version['url'])
            self.assertEqual(v1_url, disc.url_for(ver))
    self.assertIsNone(disc.url_for('v3'))
    self.assertIsNone(disc.versioned_url_for(min_version='v3', max_version='v3.latest'))
    self.assertIsNone(disc.url_for('v2.3'))
    self.assertIsNone(disc.versioned_url_for(min_version='v2.3', max_version='v2.latest'))
    self.assertTrue(mock.called_once)
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
def test_cinder_version_data(self):
    mock = self.requests_mock.get(BASE_URL, status_code=300, json=CINDER_EXAMPLES)
    disc = discover.Discover(self.session, BASE_URL)
    raw_data = disc.raw_version_data()
    clean_data = disc.version_data()
    self.assertEqual(3, len(raw_data))
    for v in raw_data:
        self.assertEqual(v['status'], discover.Status.CURRENT)
        if v['id'] == 'v1.0':
            self.assertEqual(v['updated'], '2012-01-04T11:33:21Z')
        elif v['id'] == 'v2.0':
            self.assertEqual(v['updated'], '2012-11-21T11:33:21Z')
        elif v['id'] == 'v3.0':
            self.assertEqual(v['updated'], '2012-11-21T11:33:21Z')
        else:
            self.fail('Invalid version found')
    v1_url = '%sv1/' % BASE_URL
    v2_url = '%sv2/' % BASE_URL
    v3_url = '%sv3/' % BASE_URL
    self.assertEqual(clean_data, [{'collection': None, 'max_microversion': None, 'min_microversion': None, 'next_min_version': None, 'not_before': None, 'version': (1, 0), 'url': v1_url, 'status': discover.Status.CURRENT, 'raw_status': discover.Status.CURRENT}, {'collection': None, 'max_microversion': None, 'min_microversion': None, 'next_min_version': None, 'not_before': None, 'version': (2, 0), 'url': v2_url, 'status': discover.Status.CURRENT, 'raw_status': discover.Status.CURRENT}, {'collection': BASE_URL, 'max_microversion': (3, 27), 'min_microversion': (3, 0), 'next_min_version': (3, 4), 'not_before': u'2019-12-31', 'version': (3, 0), 'url': v3_url, 'status': discover.Status.CURRENT, 'raw_status': discover.Status.CURRENT}])
    for version in (disc.data_for('v2.0'), disc.versioned_data_for(min_version='v2.0', max_version='v2.latest')):
        self.assertEqual((2, 0), version['version'])
        self.assertEqual(discover.Status.CURRENT, version['raw_status'])
        self.assertEqual(v2_url, version['url'])
    for version in (disc.data_for(1), disc.versioned_data_for(min_version=(1,), max_version=(1, discover.LATEST))):
        self.assertEqual((1, 0), version['version'])
        self.assertEqual(discover.Status.CURRENT, version['raw_status'])
        self.assertEqual(v1_url, version['url'])
    self.assertIsNone(disc.url_for('v4'))
    self.assertIsNone(disc.versioned_url_for(min_version='v4', max_version='v4.latest'))
    self.assertEqual(v3_url, disc.url_for('v3'))
    self.assertEqual(v3_url, disc.versioned_url_for(min_version='v3', max_version='v3.latest'))
    self.assertEqual(v2_url, disc.url_for('v2'))
    self.assertEqual(v2_url, disc.versioned_url_for(min_version='v2', max_version='v2.latest'))
    self.assertEqual(v1_url, disc.url_for('v1'))
    self.assertEqual(v1_url, disc.versioned_url_for(min_version='v1', max_version='v1.latest'))
    self.assertTrue(mock.called_once)
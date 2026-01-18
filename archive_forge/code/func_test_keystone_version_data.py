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
def test_keystone_version_data(self):
    mock = self.requests_mock.get(BASE_URL, status_code=300, json=V3_VERSION_LIST)
    disc = discover.Discover(self.session, BASE_URL)
    raw_data = disc.raw_version_data()
    clean_data = disc.version_data()
    self.assertEqual(2, len(raw_data))
    self.assertEqual(2, len(clean_data))
    for v in raw_data:
        self.assertIn(v['id'], ('v2.0', 'v3.0'))
        self.assertEqual(v['updated'], UPDATED)
        self.assertEqual(v['status'], 'stable')
        if v['id'] == 'v3.0':
            self.assertEqual(v['media-types'], V3_MEDIA_TYPES)
    for v in clean_data:
        self.assertIn(v['version'], ((2, 0), (3, 0)))
        self.assertEqual(v['raw_status'], 'stable')
    valid_v3_versions = (disc.data_for('v3.0'), disc.data_for('3.latest'), disc.data_for('latest'), disc.versioned_data_for(min_version='v3.0', max_version='v3.latest'), disc.versioned_data_for(min_version='3'), disc.versioned_data_for(min_version='3.latest'), disc.versioned_data_for(min_version='latest'), disc.versioned_data_for(min_version='3.latest', max_version='latest'), disc.versioned_data_for(min_version='latest', max_version='latest'), disc.versioned_data_for(min_version=2), disc.versioned_data_for(min_version='2.latest'))
    for version in valid_v3_versions:
        self.assertEqual((3, 0), version['version'])
        self.assertEqual('stable', version['raw_status'])
        self.assertEqual(V3_URL, version['url'])
    valid_v2_versions = (disc.data_for(2), disc.data_for('2.latest'), disc.versioned_data_for(min_version=2, max_version=(2, discover.LATEST)), disc.versioned_data_for(min_version='2.latest', max_version='2.latest'))
    for version in valid_v2_versions:
        self.assertEqual((2, 0), version['version'])
        self.assertEqual('stable', version['raw_status'])
        self.assertEqual(V2_URL, version['url'])
    self.assertIsNone(disc.url_for('v4'))
    self.assertIsNone(disc.versioned_url_for(min_version='v4', max_version='v4.latest'))
    self.assertEqual(V3_URL, disc.url_for('v3'))
    self.assertEqual(V3_URL, disc.versioned_url_for(min_version='v3', max_version='v3.latest'))
    self.assertEqual(V2_URL, disc.url_for('v2'))
    self.assertEqual(V2_URL, disc.versioned_url_for(min_version='v2', max_version='v2.latest'))
    self.assertTrue(mock.called_once)
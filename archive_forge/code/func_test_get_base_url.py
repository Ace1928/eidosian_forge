import json
import logging
from unittest import mock
import ddt
import fixtures
from keystoneauth1 import adapter
from keystoneauth1 import exceptions as keystone_exception
from oslo_serialization import jsonutils
from cinderclient import api_versions
import cinderclient.client
from cinderclient import exceptions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
@mock.patch('cinderclient.client.SessionClient.get_endpoint')
@ddt.data(('http://192.168.1.1:8776/v2', 'http://192.168.1.1:8776/'), ('http://192.168.1.1:8776/v3/e5526285ebd741b1819393f772f11fc3', 'http://192.168.1.1:8776/'), ('https://192.168.1.1:8080/volumes/v3/e5526285ebd741b1819393f772f11fc3', 'https://192.168.1.1:8080/volumes/'), ('http://192.168.1.1/volumes/v3/e5526285ebd741b1819393f772f11fc3', 'http://192.168.1.1/volumes/'), ('https://volume.example.com/', 'https://volume.example.com/'))
@ddt.unpack
def test_get_base_url(self, url, expected_base, mock_get_endpoint):
    mock_get_endpoint.return_value = url
    cs = cinderclient.client.SessionClient(self, api_version='3.0')
    self.assertEqual(expected_base, cs._get_base_url())
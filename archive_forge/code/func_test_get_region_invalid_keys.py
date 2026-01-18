import argparse
import copy
import os
from unittest import mock
import fixtures
import testtools
import yaml
from openstack import config
from openstack.config import cloud_region
from openstack.config import defaults
from openstack import exceptions
from openstack.tests.unit.config import base
def test_get_region_invalid_keys(self):
    invalid_conf = base._write_yaml({'clouds': {'_test_cloud': {'profile': '_test_cloud_in_our_cloud', 'auth': {'auth_url': 'http://example.com/v2', 'username': 'testuser', 'password': 'testpass'}, 'regions': [{'name': 'region1', 'foo': 'bar'}]}}})
    c = config.OpenStackConfig(config_files=[invalid_conf], vendor_files=[self.vendor_yaml])
    self.assertRaises(exceptions.ConfigException, c._get_region, cloud='_test_cloud', region_name='region1')
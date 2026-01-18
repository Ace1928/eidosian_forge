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
def test_get_one_dash_kwargs(self):
    c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
    args = {'auth-url': 'http://example.com/v2', 'username': 'user', 'password': 'password', 'project_name': 'project', 'region_name': 'other-test-region', 'snack_type': 'cookie'}
    cc = c.get_one(**args)
    self.assertIsNone(cc.cloud)
    self.assertEqual(cc.region_name, 'other-test-region')
    self.assertEqual(cc.snack_type, 'cookie')
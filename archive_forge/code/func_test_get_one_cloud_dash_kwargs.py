import argparse
import copy
import os
import extras
import fixtures
import testtools
import yaml
from openstack.config import defaults
from os_client_config import cloud_config
from os_client_config import config
from os_client_config import exceptions
from os_client_config.tests import base
def test_get_one_cloud_dash_kwargs(self):
    c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
    args = {'auth-url': 'http://example.com/v2', 'username': 'user', 'password': 'password', 'project_name': 'project', 'region_name': 'other-test-region', 'snack_type': 'cookie'}
    cc = c.get_one_cloud(**args)
    self.assertIsNone(cc.cloud)
    self.assertEqual(cc.region_name, 'other-test-region')
    self.assertEqual(cc.snack_type, 'cookie')
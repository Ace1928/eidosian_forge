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
def test_get_one_cloud_precedence_no_argparse(self):
    c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
    kwargs = {'auth': {'username': 'testuser', 'password': 'authpass', 'project-id': 'testproject', 'auth_url': 'http://example.com/v2'}, 'region_name': 'kwarg_region', 'password': 'ansible_password', 'arbitrary': 'value'}
    cc = c.get_one_cloud(**kwargs)
    self.assertEqual(cc.region_name, 'kwarg_region')
    self.assertEqual(cc.auth['password'], 'authpass')
    self.assertIsNone(cc.password)
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
def test_get_one_precedence_no_argparse(self):
    c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
    kwargs = {'auth': {'username': 'testuser', 'password': 'authpass', 'project-id': 'testproject', 'auth_url': 'http://example.com/v2'}, 'region_name': 'kwarg_region', 'password': 'ansible_password', 'arbitrary': 'value'}
    cc = c.get_one(**kwargs)
    self.assertEqual(cc.region_name, 'kwarg_region')
    self.assertEqual(cc.auth['password'], 'authpass')
    self.assertIsNone(cc.password)
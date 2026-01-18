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
def test_get_one_cloud_precedence(self):
    c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
    kwargs = {'auth': {'username': 'testuser', 'password': 'authpass', 'project-id': 'testproject', 'auth_url': 'http://example.com/v2'}, 'region_name': 'kwarg_region', 'password': 'ansible_password', 'arbitrary': 'value'}
    args = dict(auth_url='http://example.com/v2', username='user', password='argpass', project_name='project', region_name='region2', snack_type='cookie')
    options = argparse.Namespace(**args)
    cc = c.get_one_cloud(argparse=options, **kwargs)
    self.assertEqual(cc.region_name, 'region2')
    self.assertEqual(cc.auth['password'], 'authpass')
    self.assertEqual(cc.snack_type, 'cookie')
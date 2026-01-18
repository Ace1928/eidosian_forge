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
def test_get_one_precedence(self):
    c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
    kwargs = {'auth': {'username': 'testuser', 'password': 'authpass', 'project-id': 'testproject', 'auth_url': 'http://example.com/v2'}, 'region_name': 'kwarg_region', 'password': 'ansible_password', 'arbitrary': 'value'}
    args = dict(auth_url='http://example.com/v2', username='user', password='argpass', project_name='project', region_name='region2', snack_type='cookie')
    options = argparse.Namespace(**args)
    cc = c.get_one(argparse=options, **kwargs)
    self.assertEqual(cc.region_name, 'region2')
    self.assertEqual(cc.auth['password'], 'authpass')
    self.assertEqual(cc.snack_type, 'cookie')
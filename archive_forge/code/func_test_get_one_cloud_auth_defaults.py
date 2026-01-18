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
def test_get_one_cloud_auth_defaults(self):
    c = config.OpenStackConfig(config_files=[self.cloud_yaml])
    cc = c.get_one_cloud(cloud='_test-cloud_', auth={'username': 'user'})
    self.assertEqual('user', cc.auth['username'])
    self.assertEqual(defaults._defaults['auth_type'], cc.auth_type)
    self.assertEqual(defaults._defaults['identity_api_version'], cc.identity_api_version)
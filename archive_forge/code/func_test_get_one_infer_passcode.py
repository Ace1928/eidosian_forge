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
def test_get_one_infer_passcode(self):
    single_conf = base._write_yaml({'clouds': {'mfa': {'auth_type': 'v3multifactor', 'auth_methods': ['v3password', 'v3totp'], 'auth': {'auth_url': 'fake_url', 'username': 'testuser', 'password': 'testpass', 'project_name': 'testproject', 'project_domain_name': 'projectdomain', 'user_domain_name': 'udn'}, 'region_name': 'test-region'}}})
    c = config.OpenStackConfig(config_files=[single_conf])
    cc = c.get_one(cloud='mfa', passcode='123')
    self.assertEqual('123', cc.auth['passcode'])
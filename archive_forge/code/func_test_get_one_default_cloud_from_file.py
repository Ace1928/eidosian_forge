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
def test_get_one_default_cloud_from_file(self):
    single_conf = base._write_yaml({'clouds': {'single': {'auth': {'auth_url': 'http://example.com/v2', 'username': 'testuser', 'password': 'testpass', 'project_name': 'testproject'}, 'region_name': 'test-region'}}})
    c = config.OpenStackConfig(config_files=[single_conf], secure_files=[], vendor_files=[self.vendor_yaml])
    cc = c.get_one()
    self.assertEqual(cc.name, 'single')
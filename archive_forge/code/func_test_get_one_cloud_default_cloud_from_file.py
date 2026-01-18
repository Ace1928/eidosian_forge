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
def test_get_one_cloud_default_cloud_from_file(self):
    single_conf = base._write_yaml({'clouds': {'single': {'auth': {'auth_url': 'http://example.com/v2', 'username': 'testuser', 'password': 'testpass', 'project_name': 'testproject'}, 'region_name': 'test-region'}}})
    c = config.OpenStackConfig(config_files=[single_conf], vendor_files=[self.vendor_yaml])
    cc = c.get_one_cloud()
    self.assertEqual(cc.name, 'single')
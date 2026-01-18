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
def test_get_one_unscoped_identity(self):
    single_conf = base._write_yaml({'clouds': {'unscoped': {'auth': {'auth_url': 'http://example.com/v2', 'username': 'testuser', 'password': 'testpass'}}}})
    c = config.OpenStackConfig(config_files=[single_conf], secure_files=[], vendor_files=[self.vendor_yaml])
    cc = c.get_one()
    self.assertEqual('http://example.com/v2', cc.get_endpoint('identity'))
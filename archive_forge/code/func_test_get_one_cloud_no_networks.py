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
def test_get_one_cloud_no_networks(self):
    c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
    cc = c.get_one_cloud('_test-cloud-domain-scoped_')
    self.assertEqual([], cc.get_external_networks())
    self.assertEqual([], cc.get_internal_networks())
    self.assertIsNone(cc.get_nat_source())
    self.assertIsNone(cc.get_nat_destination())
    self.assertIsNone(cc.get_default_network())
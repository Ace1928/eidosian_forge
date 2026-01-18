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
def test_get_one_cloud_network(self):
    c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
    cc = c.get_one_cloud(cloud='_test_cloud_regions', region_name='region1', argparse=None)
    self._assert_cloud_details(cc)
    self.assertEqual(cc.region_name, 'region1')
    self.assertEqual('region1-network', cc.config['external_network'])
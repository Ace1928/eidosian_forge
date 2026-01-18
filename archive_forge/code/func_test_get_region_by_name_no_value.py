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
def test_get_region_by_name_no_value(self):
    c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
    region = c._get_region(cloud='_test_cloud_regions', region_name='region-no-value')
    self.assertEqual(region, {'name': 'region-no-value', 'values': {}})
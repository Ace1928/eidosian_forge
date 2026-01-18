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
def test_get_one_cloud_osc_password_brace(self):
    c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
    password = 'foo{'
    self.options.password = password
    cc = c.get_one_cloud_osc(cloud='_test_cloud_regions', argparse=self.options, validate=False)
    self.assertEqual(cc.password, password)
    password = 'foo{bar}'
    self.options.password = password
    cc = c.get_one_cloud_osc(cloud='_test_cloud_regions', argparse=self.options, validate=False)
    self.assertEqual(cc.password, password)
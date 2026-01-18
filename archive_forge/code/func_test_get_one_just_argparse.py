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
def test_get_one_just_argparse(self):
    c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
    cc = c.get_one(argparse=self.options, validate=False)
    self.assertIsNone(cc.cloud)
    self.assertEqual(cc.region_name, 'region2')
    self.assertEqual(cc.snack_type, 'cookie')
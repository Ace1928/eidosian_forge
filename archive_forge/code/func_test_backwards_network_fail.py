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
def test_backwards_network_fail(self):
    c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
    cloud = {'external_network': 'public', 'networks': [{'name': 'private', 'routes_externally': False}]}
    self.assertRaises(exceptions.ConfigException, c._fix_backwards_networks, cloud)
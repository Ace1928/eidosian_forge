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
def test_get_cloud_names(self):
    c = config.OpenStackConfig(config_files=[self.cloud_yaml], secure_files=[self.no_yaml])
    self.assertCountEqual(['_test-cloud-domain-id_', '_test-cloud-domain-scoped_', '_test-cloud-int-project_', '_test-cloud-networks_', '_test-cloud_', '_test-cloud_no_region', '_test_cloud_hyphenated', '_test_cloud_no_vendor', '_test_cloud_regions', '_test-cloud-override-metrics'], c.get_cloud_names())
    c = config.OpenStackConfig(config_files=[self.no_yaml], vendor_files=[self.no_yaml], secure_files=[self.no_yaml])
    for k in os.environ.keys():
        if k.startswith('OS_'):
            self.useFixture(fixtures.EnvironmentVariable(k))
    c.get_one(cloud='defaults', validate=False)
    self.assertEqual(['defaults'], sorted(c.get_cloud_names()))
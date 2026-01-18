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
def test_get_one_domain_scoped(self):
    c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
    cc = c.get_one('_test-cloud-domain-scoped_')
    self.assertEqual('12345', cc.auth['domain_id'])
    self.assertNotIn('user_domain_id', cc.auth)
    self.assertNotIn('project_domain_id', cc.auth)
    self.assertIsNone(cc.get_endpoint('identity'))
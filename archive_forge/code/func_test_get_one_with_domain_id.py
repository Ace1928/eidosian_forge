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
def test_get_one_with_domain_id(self):
    c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
    cc = c.get_one('_test-cloud-domain-id_')
    self.assertEqual('6789', cc.auth['user_domain_id'])
    self.assertEqual('123456789', cc.auth['project_domain_id'])
    self.assertNotIn('domain_id', cc.auth)
    self.assertNotIn('domain-id', cc.auth)
    self.assertNotIn('domain_id', cc)
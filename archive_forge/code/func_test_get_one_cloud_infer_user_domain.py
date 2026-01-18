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
def test_get_one_cloud_infer_user_domain(self):
    c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
    cc = c.get_one_cloud('_test-cloud-int-project_')
    self.assertEqual('awesome-domain', cc.auth['user_domain_id'])
    self.assertEqual('awesome-domain', cc.auth['project_domain_id'])
    self.assertNotIn('domain_id', cc.auth)
    self.assertNotIn('domain_id', cc)
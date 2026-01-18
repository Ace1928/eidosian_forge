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
def test_get_one_cloud_no_yaml(self):
    c = config.OpenStackConfig(load_yaml_config=False)
    cc = c.get_one_cloud(region_name='region2', argparse=None, **base.USER_CONF['clouds']['_test_cloud_regions'])
    self.assertIsInstance(cc, cloud_config.CloudConfig)
    self.assertTrue(extras.safe_hasattr(cc, 'auth'))
    self.assertIsInstance(cc.auth, dict)
    self.assertIsNone(cc.cloud)
    self.assertIn('username', cc.auth)
    self.assertEqual('testuser', cc.auth['username'])
    self.assertEqual('testpass', cc.auth['password'])
    self.assertFalse(cc.config['image_api_use_tasks'])
    self.assertTrue('project_name' in cc.auth or 'project_id' in cc.auth)
    if 'project_name' in cc.auth:
        self.assertEqual('testproject', cc.auth['project_name'])
    elif 'project_id' in cc.auth:
        self.assertEqual('testproject', cc.auth['project_id'])
    self.assertEqual(cc.region_name, 'region2')
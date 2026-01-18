import os
from unittest import mock
import fixtures
from keystoneauth1 import session
from testtools import matchers
import openstack.config
from openstack import connection
from openstack import proxy
from openstack import service_description
from openstack.tests import fakes
from openstack.tests.unit import base
from openstack.tests.unit.fake import fake_service
def test_from_config_given_cloud_config(self):
    cloud_region = openstack.config.OpenStackConfig().get_one('sample-cloud')
    sot = connection.from_config(cloud_config=cloud_region)
    self.assertEqual(CONFIG_USERNAME, sot.config.config['auth']['username'])
    self.assertEqual(CONFIG_PASSWORD, sot.config.config['auth']['password'])
    self.assertEqual(CONFIG_AUTH_URL, sot.config.config['auth']['auth_url'])
    self.assertEqual(CONFIG_PROJECT, sot.config.config['auth']['project_name'])
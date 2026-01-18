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
def test_hook_from_profile(self):
    self.cloud = self.config.get_one(cloud='profiled-cloud')
    conn = connection.Connection(config=self.cloud)
    self.assertEqual('test_val', conn.test)
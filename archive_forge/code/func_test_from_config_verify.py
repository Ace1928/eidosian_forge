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
def test_from_config_verify(self):
    sot = connection.from_config(cloud='insecure-cloud')
    self.assertFalse(sot.session.verify)
    sot = connection.from_config(cloud='cacert-cloud')
    self.assertEqual(CONFIG_CACERT, sot.session.verify)
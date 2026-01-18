from unittest import mock
from openstack.cloud import meta
from openstack.compute.v2 import server as _server
from openstack import connection
from openstack.tests import fakes
from openstack.tests.unit import base
def test_current_project(self):
    self.assertEqual({'id': mock.ANY, 'name': 'admin', 'domain_id': None, 'domain_name': 'default'}, self.cloud.current_project)
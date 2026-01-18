from unittest import mock
from openstack.cloud import meta
from openstack.compute.v2 import server as _server
from openstack import connection
from openstack.tests import fakes
from openstack.tests.unit import base
def test_get_groups_from_server(self):
    server_vars = {'flavor': 'test-flavor', 'image': 'test-image', 'az': 'test-az'}
    self.assertEqual(['test-name', 'test-region', 'test-name_test-region', 'test-group', 'instance-test-id-0', 'meta-group_test-group', 'test-az', 'test-region_test-az', 'test-name_test-region_test-az'], meta.get_groups_from_server(FakeCloud(), meta.obj_to_munch(standard_fake_server), server_vars))
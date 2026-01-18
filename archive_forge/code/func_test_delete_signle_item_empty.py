from unittest import mock
from keystoneauth1 import adapter
from openstack.common import metadata
from openstack import exceptions
from openstack import resource
from openstack.tests.unit import base
from openstack.tests.unit.test_resource import FakeResponse
def test_delete_signle_item_empty(self):
    res = self.sot
    result = res.delete_metadata_item(self.session, 'foo2')
    self.assertEqual({}, res.metadata)
    self.assertEqual(res, result)
    url = self.base_path + '/' + res.id + '/metadata/foo2'
    self.session.delete.assert_called_once_with(url)
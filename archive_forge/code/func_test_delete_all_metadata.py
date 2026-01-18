from unittest import mock
from keystoneauth1 import adapter
from openstack.common import metadata
from openstack import exceptions
from openstack import resource
from openstack.tests.unit import base
from openstack.tests.unit.test_resource import FakeResponse
def test_delete_all_metadata(self):
    res = self.sot
    res.metadata = {'foo': 'bar'}
    result = res.delete_metadata(self.session)
    self.assertEqual({}, res.metadata)
    self.assertEqual(res, result)
    url = self.base_path + '/' + res.id + '/metadata'
    self.session.put.assert_called_once_with(url, json={'metadata': {}})
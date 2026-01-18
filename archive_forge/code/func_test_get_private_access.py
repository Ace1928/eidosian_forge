from unittest import mock
from keystoneauth1 import adapter
from openstack.block_storage.v2 import type
from openstack.tests.unit import base
def test_get_private_access(self):
    sot = type.Type(**TYPE)
    response = mock.Mock()
    response.status_code = 200
    response.body = {'volume_type_access': [{'project_id': 'a', 'volume_type_id': 'b'}]}
    response.json = mock.Mock(return_value=response.body)
    self.sess.get = mock.Mock(return_value=response)
    self.assertEqual(response.body['volume_type_access'], sot.get_private_access(self.sess))
    self.sess.get.assert_called_with('types/%s/os-volume-type-access' % sot.id)
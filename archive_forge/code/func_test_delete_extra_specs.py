from unittest import mock
from keystoneauth1 import adapter
from openstack.block_storage.v3 import type
from openstack import exceptions
from openstack.tests.unit import base
def test_delete_extra_specs(self):
    sess = mock.Mock()
    response = mock.Mock()
    response.status_code = 200
    sess.delete.return_value = response
    sot = type.Type(id=FAKE_ID)
    key = 'hey'
    sot.delete_extra_specs(sess, [key])
    sess.delete.assert_called_once_with('types/' + FAKE_ID + '/extra_specs/' + key, headers={})
from unittest import mock
from keystoneauth1 import adapter
from openstack.block_storage.v3 import group_type
from openstack.tests.unit import base
def test_delete_group_specs_property(self):
    sot = group_type.GroupType(**GROUP_TYPE)
    resp = mock.Mock()
    resp.body = None
    resp.json = mock.Mock(return_value=resp.body)
    resp.status_code = 200
    self.sess.delete = mock.Mock(return_value=resp)
    rsp = sot.delete_group_specs_property(self.sess, 'a')
    self.sess.delete.assert_called_with(f'group_types/{GROUP_TYPE['id']}/group_specs/a', microversion=self.sess.default_microversion)
    self.assertIsNone(rsp)
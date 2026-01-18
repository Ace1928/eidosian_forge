from unittest import mock
from keystoneauth1 import adapter
from openstack.block_storage.v3 import group_type
from openstack.tests.unit import base
def test_create_group_specs(self):
    sot = group_type.GroupType(**GROUP_TYPE)
    specs = {'a': 'b', 'c': 'd'}
    resp = mock.Mock()
    resp.body = {'group_specs': specs}
    resp.json = mock.Mock(return_value=resp.body)
    resp.status_code = 200
    self.sess.post = mock.Mock(return_value=resp)
    rsp = sot.create_group_specs(self.sess, specs)
    self.sess.post.assert_called_with(f'group_types/{GROUP_TYPE['id']}/group_specs', json={'group_specs': specs}, microversion=self.sess.default_microversion)
    self.assertEqual(resp.body['group_specs'], rsp.group_specs)
    self.assertIsInstance(rsp, group_type.GroupType)
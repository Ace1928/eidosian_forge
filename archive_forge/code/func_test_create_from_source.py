import copy
from unittest import mock
from keystoneauth1 import adapter
from openstack.block_storage.v3 import group
from openstack.tests.unit import base
def test_create_from_source(self):
    resp = mock.Mock()
    resp.body = {'group': copy.deepcopy(GROUP)}
    resp.json = mock.Mock(return_value=resp.body)
    resp.headers = {}
    resp.status_code = 202
    self.sess.post = mock.Mock(return_value=resp)
    sot = group.Group.create_from_source(self.sess, group_snapshot_id='9a591346-e595-4bc1-94e7-08f264406b63', source_group_id='6c5259f6-42ed-4e41-8ffe-e1c667ae9dff', name='group_from_source', description='a group from source')
    self.assertIsNotNone(sot)
    url = 'groups/action'
    body = {'create-from-src': {'name': 'group_from_source', 'description': 'a group from source', 'group_snapshot_id': '9a591346-e595-4bc1-94e7-08f264406b63', 'source_group_id': '6c5259f6-42ed-4e41-8ffe-e1c667ae9dff'}}
    self.sess.post.assert_called_with(url, json=body, microversion=sot._max_microversion)
from unittest import mock
from keystoneauth1 import adapter
from openstack.block_storage.v2 import type
from openstack.tests.unit import base
def test_add_private_access(self):
    sot = type.Type(**TYPE)
    self.assertIsNone(sot.add_private_access(self.sess, 'a'))
    url = 'types/%s/action' % sot.id
    body = {'addProjectAccess': {'project': 'a'}}
    self.sess.post.assert_called_with(url, json=body)
from unittest import mock
from keystoneauth1 import adapter
from openstack.shared_file_system.v2 import share
from openstack.tests.unit import base
def test_extend_share(self):
    sot = share.Share(**EXAMPLE)
    microversion = sot._get_microversion(self.sess, action='patch')
    self.assertIsNone(sot.extend_share(self.sess, new_size=3))
    url = f'shares/{IDENTIFIER}/action'
    body = {'extend': {'new_size': 3}}
    headers = {'Accept': ''}
    self.sess.post.assert_called_with(url, json=body, headers=headers, microversion=microversion)
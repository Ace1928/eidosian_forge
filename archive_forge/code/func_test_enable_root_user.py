from unittest import mock
from openstack.database.v1 import instance
from openstack.tests.unit import base
def test_enable_root_user(self):
    sot = instance.Instance(**EXAMPLE)
    response = mock.Mock()
    response.body = {'user': {'name': 'root', 'password': 'foo'}}
    response.json = mock.Mock(return_value=response.body)
    sess = mock.Mock()
    sess.post = mock.Mock(return_value=response)
    self.assertEqual(response.body['user'], sot.enable_root_user(sess))
    url = 'instances/%s/root' % IDENTIFIER
    sess.post.assert_called_with(url)
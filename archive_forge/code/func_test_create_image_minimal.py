from unittest import mock
from openstack.compute.v2 import flavor
from openstack.compute.v2 import server
from openstack.image.v2 import image
from openstack.tests.unit import base
def test_create_image_minimal(self):
    sot = server.Server(**EXAMPLE)
    name = 'noo'
    url = 'servers/IDENTIFIER/action'
    body = {'createImage': {'name': name}}
    headers = {'Accept': ''}
    rsp = mock.Mock()
    rsp.json.return_value = None
    rsp.headers = {'Location': 'dummy/dummy2'}
    rsp.status_code = 200
    self.sess.post.return_value = rsp
    self.endpoint_data = mock.Mock(spec=['min_microversion', 'max_microversion'], min_microversion='2.1', max_microversion='2.56')
    self.sess.get_endpoint_data.return_value = self.endpoint_data
    self.sess.default_microversion = None
    self.assertIsNone(self.resp.body, sot.create_image(self.sess, name))
    self.sess.post.assert_called_with(url, json=body, headers=headers, microversion='2.45')
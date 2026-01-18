from unittest import mock
from openstack.compute.v2 import flavor
from openstack.compute.v2 import server
from openstack.image.v2 import image
from openstack.tests.unit import base
def test_add_floating_ip_with_fixed_addr(self):
    sot = server.Server(**EXAMPLE)
    res = sot.add_floating_ip(self.sess, 'FLOATING-IP', 'FIXED-ADDR')
    self.assertIsNone(res)
    url = 'servers/IDENTIFIER/action'
    body = {'addFloatingIp': {'address': 'FLOATING-IP', 'fixed_address': 'FIXED-ADDR'}}
    headers = {'Accept': ''}
    self.sess.post.assert_called_with(url, json=body, headers=headers, microversion=self.sess.default_microversion)
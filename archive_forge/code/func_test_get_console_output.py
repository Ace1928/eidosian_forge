from unittest import mock
from openstack.compute.v2 import flavor
from openstack.compute.v2 import server
from openstack.image.v2 import image
from openstack.tests.unit import base
def test_get_console_output(self):
    sot = server.Server(**EXAMPLE)
    res = sot.get_console_output(self.sess)
    self.assertIsNone(res)
    url = 'servers/IDENTIFIER/action'
    body = {'os-getConsoleOutput': {}}
    headers = {'Accept': ''}
    self.sess.post.assert_called_with(url, json=body, headers=headers, microversion=self.sess.default_microversion)
    res = sot.get_console_output(self.sess, length=1)
    self.assertIsNone(res)
    url = 'servers/IDENTIFIER/action'
    body = {'os-getConsoleOutput': {'length': 1}}
    headers = {'Accept': ''}
    self.sess.post.assert_called_with(url, json=body, headers=headers, microversion=self.sess.default_microversion)
from unittest import mock
from openstack.compute.v2 import flavor
from openstack.compute.v2 import server
from openstack.image.v2 import image
from openstack.tests.unit import base
def test_evacuate_with_options(self):
    sot = server.Server(**EXAMPLE)
    res = sot.evacuate(self.sess, host='HOST2', admin_pass='NEW_PASS', force=True)
    self.assertIsNone(res)
    url = 'servers/IDENTIFIER/action'
    body = {'evacuate': {'host': 'HOST2', 'adminPass': 'NEW_PASS', 'force': True}}
    headers = {'Accept': ''}
    self.sess.post.assert_called_with(url, json=body, headers=headers, microversion=self.sess.default_microversion)
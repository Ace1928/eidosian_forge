from unittest import mock
from keystoneauth1 import adapter
from openstack.dns.v2 import zone
from openstack.tests.unit import base
def test_abandon(self):
    sot = zone.Zone(**EXAMPLE)
    self.assertIsNone(sot.abandon(self.sess))
    self.sess.post.assert_called_with('zones/NAME/tasks/abandon', json=None)
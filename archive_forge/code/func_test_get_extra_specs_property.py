from unittest import mock
from keystoneauth1 import adapter
from openstack.compute.v2 import flavor
from openstack.tests.unit import base
def test_get_extra_specs_property(self):
    sot = flavor.Flavor(**BASIC_EXAMPLE)
    resp = mock.Mock()
    resp.body = {'a': 'b'}
    resp.json = mock.Mock(return_value=resp.body)
    resp.status_code = 200
    self.sess.get = mock.Mock(return_value=resp)
    rsp = sot.get_extra_specs_property(self.sess, 'a')
    self.sess.get.assert_called_with('flavors/IDENTIFIER/os-extra_specs/a', microversion=self.sess.default_microversion)
    self.assertEqual('b', rsp)
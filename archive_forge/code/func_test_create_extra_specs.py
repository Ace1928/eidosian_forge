from unittest import mock
from keystoneauth1 import adapter
from openstack.compute.v2 import flavor
from openstack.tests.unit import base
def test_create_extra_specs(self):
    sot = flavor.Flavor(**BASIC_EXAMPLE)
    specs = {'a': 'b', 'c': 'd'}
    resp = mock.Mock()
    resp.body = {'extra_specs': specs}
    resp.json = mock.Mock(return_value=resp.body)
    resp.status_code = 200
    self.sess.post = mock.Mock(return_value=resp)
    rsp = sot.create_extra_specs(self.sess, specs)
    self.sess.post.assert_called_with('flavors/IDENTIFIER/os-extra_specs', json={'extra_specs': specs}, microversion=self.sess.default_microversion)
    self.assertEqual(resp.body['extra_specs'], rsp.extra_specs)
    self.assertIsInstance(rsp, flavor.Flavor)
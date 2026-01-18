from openstack import format as _format
from openstack import resource
from openstack.test import fakes
from openstack.tests.unit import base
def test_generate_fake_resource_attrs(self):

    class Fake(resource.Resource):
        a = resource.Body('a', type=str)
        b = resource.Body('b', type=str)
    res = fakes.generate_fake_resource(Fake, b='bar')
    self.assertIsInstance(res.a, str)
    self.assertIsInstance(res.b, str)
    self.assertEqual('bar', res.b)
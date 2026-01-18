from openstack import format as _format
from openstack import resource
from openstack.test import fakes
from openstack.tests.unit import base
def test_generate_fake_resource_types(self):

    class Foo(resource.Resource):
        a = resource.Body('a', type=str)
        b = resource.Body('b', type=int)
        c = resource.Body('c', type=bool)
        d = resource.Body('d', type=_format.BoolStr)
        e = resource.Body('e', type=dict)
        f = resource.URI('path')

    class Bar(resource.Resource):
        a = resource.Body('a', type=list, list_type=str)
        b = resource.Body('b', type=list, list_type=dict)
        c = resource.Body('c', type=list, list_type=Foo)
    foo = fakes.generate_fake_resource(Foo)
    self.assertIsInstance(foo.a, str)
    self.assertIsInstance(foo.b, int)
    self.assertIsInstance(foo.c, bool)
    self.assertIsInstance(foo.d, bool)
    self.assertIsInstance(foo.e, dict)
    self.assertIsInstance(foo.f, str)
    bar = fakes.generate_fake_resource(Bar)
    self.assertIsInstance(bar.a, list)
    self.assertEqual(1, len(bar.a))
    self.assertIsInstance(bar.a[0], str)
    self.assertIsInstance(bar.b, list)
    self.assertEqual(1, len(bar.b))
    self.assertIsInstance(bar.b[0], dict)
    self.assertIsInstance(bar.c, list)
    self.assertEqual(1, len(bar.c))
    self.assertIsInstance(bar.c[0], Foo)
    self.assertIsInstance(bar.c[0].a, str)
    self.assertIsInstance(bar.c[0].b, int)
    self.assertIsInstance(bar.c[0].c, bool)
    self.assertIsInstance(bar.c[0].d, bool)
    self.assertIsInstance(bar.c[0].e, dict)
    self.assertIsInstance(bar.c[0].f, str)
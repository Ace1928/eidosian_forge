import functools
from oslotest import base as test_base
from oslo_utils import reflection
def test_inner_class(self):
    obj = self.InnerCallableClass()
    name = reflection.get_callable_name(obj)
    expected_name = '.'.join((__name__, 'GetCallableNameTestExtended', 'InnerCallableClass'))
    self.assertEqual(expected_name, name)
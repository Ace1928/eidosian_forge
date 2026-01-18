import functools
from oslotest import base as test_base
from oslo_utils import reflection
def test_callable_class_call(self):
    name = reflection.get_callable_name(CallableClass().__call__)
    self.assertEqual('.'.join((__name__, 'CallableClass', '__call__')), name)
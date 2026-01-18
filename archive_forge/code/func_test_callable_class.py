import functools
from oslotest import base as test_base
from oslo_utils import reflection
def test_callable_class(self):
    name = reflection.get_callable_name(CallableClass())
    self.assertEqual('.'.join((__name__, 'CallableClass')), name)
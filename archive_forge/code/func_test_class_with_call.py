import functools
from oslotest import base as test_base
from oslo_utils import reflection
def test_class_with_call(self):
    result = reflection.get_callable_args(CallableClass())
    self.assertEqual(['i', 'j'], result)
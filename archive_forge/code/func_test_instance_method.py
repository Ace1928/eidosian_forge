import functools
from oslotest import base as test_base
from oslo_utils import reflection
def test_instance_method(self):
    result = reflection.get_callable_args(Class().method)
    self.assertEqual(['c', 'd'], result)
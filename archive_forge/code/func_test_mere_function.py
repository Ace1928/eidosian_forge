import functools
from oslotest import base as test_base
from oslo_utils import reflection
def test_mere_function(self):
    result = reflection.get_callable_args(mere_function)
    self.assertEqual(['a', 'b'], result)
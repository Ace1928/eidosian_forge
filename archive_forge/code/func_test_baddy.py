import functools
from oslotest import base as test_base
from oslo_utils import reflection
def test_baddy(self):
    b = BadClass()
    self.assertTrue(reflection.is_bound_method(b.do_something))
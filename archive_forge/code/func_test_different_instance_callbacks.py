import functools
from oslotest import base as test_base
from oslo_utils import reflection
def test_different_instance_callbacks(self):

    class A(object):

        def b(self):
            pass

        def __eq__(self, other):
            return True

        def __ne__(self, other):
            return not self.__eq__(other)
    b = A()
    c = A()
    self.assertFalse(reflection.is_same_callback(b.b, c.b))
    self.assertTrue(reflection.is_same_callback(b.b, b.b))
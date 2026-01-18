import functools
from oslotest import base as test_base
from oslo_utils import reflection
def test_bound_method(self):
    c = Class()
    name = reflection.get_class_name(c.method)
    self.assertEqual('%s.Class' % __name__, name)
    name = reflection.get_class_name(c.method, fully_qualified=False)
    self.assertEqual('Class', name)
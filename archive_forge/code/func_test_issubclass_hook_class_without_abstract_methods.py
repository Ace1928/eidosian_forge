import abc
from neutron_lib.services import base
from neutron_lib.tests import _base as test_base
def test_issubclass_hook_class_without_abstract_methods(self):

    class A(object):

        def f(self):
            pass

    class B(base.ServicePluginBase):

        def f(self):
            pass
    self.assertFalse(issubclass(A, B))
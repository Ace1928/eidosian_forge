import unittest
import warnings
from traits.adaptation.api import reset_global_adaptation_manager
from traits.api import (
from traits.interface_checker import InterfaceError, check_implements
from traits import has_traits
def test_non_traits_class(self):
    """ non-traits class """

    class IFoo(Interface):

        def foo(self):
            pass

    @provides(IFoo)
    class Foo(object):

        def foo(self):
            pass
    check_implements(Foo, IFoo, 2)
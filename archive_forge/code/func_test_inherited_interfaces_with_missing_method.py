import unittest
import warnings
from traits.adaptation.api import reset_global_adaptation_manager
from traits.api import (
from traits.interface_checker import InterfaceError, check_implements
from traits import has_traits
def test_inherited_interfaces_with_missing_method(self):
    """ inherited interfaces with missing method """

    class IFoo(Interface):

        def foo(self):
            pass

    class IBar(IFoo):

        def bar(self):
            pass

    class IBaz(IBar):

        def baz(self):
            pass

    @provides(IBaz)
    class Foo(HasTraits):

        def foo(self):
            pass

        def bar(self):
            pass
    self.assertRaises(InterfaceError, check_implements, Foo, IBaz, 2)
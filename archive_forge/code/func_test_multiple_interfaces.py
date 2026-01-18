import unittest
import warnings
from traits.adaptation.api import reset_global_adaptation_manager
from traits.api import (
from traits.interface_checker import InterfaceError, check_implements
from traits import has_traits
def test_multiple_interfaces(self):
    """ multiple interfaces """

    class IFoo(Interface):
        x = Int

    class IBar(Interface):
        y = Int

    class IBaz(Interface):
        z = Int

    @provides(IFoo, IBar, IBaz)
    class Foo(HasTraits):
        x = Int
        y = Int
        z = Int
    check_implements(Foo, [IFoo, IBar, IBaz], 2)
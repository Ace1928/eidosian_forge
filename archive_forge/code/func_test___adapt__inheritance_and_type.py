import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test___adapt__inheritance_and_type(self):
    from zope.interface import Interface
    from zope.interface import interfacemethod

    class IRoot(Interface):
        """Root"""

    class IWithAdapt(IRoot):

        @interfacemethod
        def __adapt__(self, obj):
            return 42

    class IOther(IRoot):
        """Second branch"""

    class IUnrelated(Interface):
        """Unrelated"""

    class IDerivedAdapt(IUnrelated, IWithAdapt, IOther):
        """Inherits an adapt"""

    class IDerived2Adapt(IDerivedAdapt):
        """Overrides an inherited custom adapt."""

        @interfacemethod
        def __adapt__(self, obj):
            return 24
    self.assertEqual(42, IDerivedAdapt(object()))
    for iface in (IRoot, IWithAdapt, IOther, IUnrelated, IDerivedAdapt):
        self.assertEqual(__name__, iface.__module__)
    for iface in (IRoot, IOther, IUnrelated):
        self.assertEqual(type(IRoot), type(Interface))
    self.assertNotEqual(type(Interface), type(IWithAdapt))
    self.assertEqual(type(IWithAdapt), type(IDerivedAdapt))
    self.assertIsInstance(IWithAdapt, type(Interface))
    self.assertEqual(24, IDerived2Adapt(object()))
    self.assertNotEqual(type(IDerived2Adapt), type(IDerivedAdapt))
    self.assertIsInstance(IDerived2Adapt, type(IDerivedAdapt))
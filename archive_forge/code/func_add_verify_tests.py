import unittest
from zope.interface.common import ABCInterface
from zope.interface.common import ABCInterfaceClass
from zope.interface.verify import verifyClass
from zope.interface.verify import verifyObject
def add_verify_tests(cls, iface_classes_iter):
    cls.maxDiff = None
    for iface, registered_classes in iface_classes_iter:
        for stdlib_class in registered_classes:

            def test(self, stdlib_class=stdlib_class, iface=iface):
                if stdlib_class in self.UNVERIFIABLE or stdlib_class.__name__ in self.UNVERIFIABLE:
                    self.skipTest('Unable to verify %s' % stdlib_class)
                self.assertTrue(self.verify(iface, stdlib_class))
            suffix = '{}_{}_{}_{}'.format(stdlib_class.__module__.replace('.', '_'), stdlib_class.__name__, iface.__module__.replace('.', '_'), iface.__name__)
            name = 'test_auto_' + suffix
            test.__name__ = name
            assert not hasattr(cls, name), (name, list(cls.__dict__))
            setattr(cls, name, test)

            def test_ro(self, stdlib_class=stdlib_class, iface=iface):
                from zope.interface import Interface
                from zope.interface import implementedBy
                from zope.interface import ro
                self.assertEqual(tuple(ro.ro(iface, strict=True)), iface.__sro__)
                implements = implementedBy(stdlib_class)
                sro = implements.__sro__
                self.assertIs(sro[-1], Interface)
                if stdlib_class not in self.UNVERIFIABLE_RO:
                    strict = stdlib_class not in self.NON_STRICT_RO
                    isro = ro.ro(implements, strict=strict)
                    isro.remove(Interface)
                    isro.append(Interface)
                    self.assertEqual(tuple(isro), sro)
            name = 'test_auto_ro_' + suffix
            test_ro.__name__ = name
            assert not hasattr(cls, name)
            setattr(cls, name, test_ro)
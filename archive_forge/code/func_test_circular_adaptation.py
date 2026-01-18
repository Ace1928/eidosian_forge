import unittest
from traits.adaptation.api import AdaptationManager
import traits.adaptation.tests.abc_examples
import traits.adaptation.tests.interface_examples
def test_circular_adaptation(self):

    class Foo(object):
        pass

    class Bar(object):
        pass
    self.adaptation_manager.register_factory(factory=lambda adaptee: Foo(), from_protocol=object, to_protocol=Foo)
    self.adaptation_manager.register_factory(factory=lambda adaptee: [], from_protocol=Foo, to_protocol=object)
    obj = []
    bar = self.adaptation_manager.adapt(obj, Bar, None)
    self.assertIsNone(bar)
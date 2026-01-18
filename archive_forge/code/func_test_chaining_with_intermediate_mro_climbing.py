import unittest
from traits.adaptation.api import AdaptationManager
import traits.adaptation.tests.abc_examples
import traits.adaptation.tests.interface_examples
def test_chaining_with_intermediate_mro_climbing(self):
    ex = self.examples
    self.adaptation_manager.register_factory(factory=ex.IStartToISpecific, from_protocol=ex.IStart, to_protocol=ex.ISpecific)
    self.adaptation_manager.register_factory(factory=ex.IGenericToIEnd, from_protocol=ex.IGeneric, to_protocol=ex.IEnd)
    start = ex.Start()
    end = self.adaptation_manager.adapt(start, ex.IEnd)
    self.assertIsNotNone(end)
    self.assertIs(type(end), ex.IGenericToIEnd)
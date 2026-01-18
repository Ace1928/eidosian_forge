import unittest
from traits.adaptation.api import (
import traits.adaptation.tests.abc_examples
def test_global_convenience_functions(self):
    ex = self.examples
    register_factory(factory=ex.UKStandardToEUStandard, from_protocol=ex.UKStandard, to_protocol=ex.EUStandard)
    uk_plug = ex.UKPlug()
    eu_plug = adapt(uk_plug, ex.EUStandard)
    self.assertIsNotNone(eu_plug)
    self.assertIsInstance(eu_plug, ex.UKStandardToEUStandard)
    self.assertTrue(provides_protocol(ex.UKPlug, ex.UKStandard))
    self.assertTrue(supports_protocol(uk_plug, ex.EUStandard))
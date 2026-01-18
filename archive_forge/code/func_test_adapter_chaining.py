import unittest
from traits.adaptation.api import AdaptationManager
import traits.adaptation.tests.abc_examples
import traits.adaptation.tests.interface_examples
def test_adapter_chaining(self):
    ex = self.examples
    self.adaptation_manager.register_factory(factory=ex.UKStandardToEUStandard, from_protocol=ex.UKStandard, to_protocol=ex.EUStandard)
    self.adaptation_manager.register_factory(factory=ex.EUStandardToJapanStandard, from_protocol=ex.EUStandard, to_protocol=ex.JapanStandard)
    uk_plug = ex.UKPlug()
    japan_plug = self.adaptation_manager.adapt(uk_plug, ex.JapanStandard)
    self.assertIsNotNone(japan_plug)
    self.assertIsInstance(japan_plug, ex.EUStandardToJapanStandard)
    self.assertIs(japan_plug.adaptee.adaptee, uk_plug)
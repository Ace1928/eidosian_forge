import unittest
from traits.adaptation.api import AdaptationManager
import traits.adaptation.tests.abc_examples
import traits.adaptation.tests.interface_examples
def test_multiple_paths_unambiguous(self):
    ex = self.examples
    self.adaptation_manager.register_factory(factory=ex.UKStandardToEUStandard, from_protocol=ex.UKStandard, to_protocol=ex.EUStandard)
    self.adaptation_manager.register_factory(factory=ex.EUStandardToJapanStandard, from_protocol=ex.EUStandard, to_protocol=ex.JapanStandard)
    self.adaptation_manager.register_factory(factory=ex.JapanStandardToIraqStandard, from_protocol=ex.JapanStandard, to_protocol=ex.IraqStandard)
    self.adaptation_manager.register_factory(factory=ex.EUStandardToIraqStandard, from_protocol=ex.EUStandard, to_protocol=ex.IraqStandard)
    uk_plug = ex.UKPlug()
    iraq_plug = self.adaptation_manager.adapt(uk_plug, ex.IraqStandard)
    self.assertIsNotNone(iraq_plug)
    self.assertIsInstance(iraq_plug, ex.EUStandardToIraqStandard)
    self.assertIs(iraq_plug.adaptee.adaptee, uk_plug)
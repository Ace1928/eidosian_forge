import unittest
from traits.adaptation.api import AdaptationManager
import traits.adaptation.tests.abc_examples
import traits.adaptation.tests.interface_examples
def test_no_adapter_required(self):
    ex = self.examples
    plug = ex.UKPlug()
    uk_plug = self.adaptation_manager.adapt(plug, ex.UKPlug)
    self.assertIs(uk_plug, plug)
    uk_plug = self.adaptation_manager.adapt(plug, ex.UKStandard)
    self.assertIs(uk_plug, plug)
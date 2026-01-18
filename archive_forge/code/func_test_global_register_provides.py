import unittest
from traits.adaptation.api import (
import traits.adaptation.tests.abc_examples
def test_global_register_provides(self):
    from traits.api import Interface

    class IFoo(Interface):
        pass
    obj = {}
    register_provides(dict, IFoo)
    self.assertEqual(obj, adapt(obj, IFoo))
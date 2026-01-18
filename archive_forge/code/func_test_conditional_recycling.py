import unittest
from traits.adaptation.api import AdaptationManager
import traits.adaptation.tests.abc_examples
import traits.adaptation.tests.interface_examples
def test_conditional_recycling(self):

    class A(object):

        def __init__(self, allow_adaptation):
            self.allow_adaptation = allow_adaptation

    class B(object):
        pass

    class C(object):
        pass

    class D(object):
        pass
    self.adaptation_manager.register_factory(factory=lambda adaptee: A(False), from_protocol=C, to_protocol=A)
    self.adaptation_manager.register_factory(factory=lambda adaptee: A(True), from_protocol=D, to_protocol=A)
    self.adaptation_manager.register_factory(factory=lambda adaptee: D(), from_protocol=C, to_protocol=D)

    def a_to_b_adapter(adaptee):
        if adaptee.allow_adaptation:
            b = B()
            b.marker = True
        else:
            b = None
        return b
    self.adaptation_manager.register_factory(factory=a_to_b_adapter, from_protocol=A, to_protocol=B)
    c = C()
    b = self.adaptation_manager.adapt(c, B)
    self.assertIsNotNone(b)
    self.assertTrue(hasattr(b, 'marker'))
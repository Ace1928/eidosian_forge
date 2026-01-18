import unittest
from traits.api import on_trait_change
from traits.adaptation.api import Adapter
@on_trait_change('adaptee')
def check_that_adaptee_change_is_notified(self):
    self.adaptee_notifier_called = True
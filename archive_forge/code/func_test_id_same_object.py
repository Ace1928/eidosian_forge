import unittest
import warnings
from traits.api import (
def test_id_same_object(self):
    ic = IdentityCompare()
    ic.on_trait_change(self.bar_changed, 'bar')
    self.reset_change_tracker()
    default_value = ic.bar
    ic.bar = self.a
    self.check_tracker(ic, 'bar', default_value, self.a, 1)
    ic.bar = self.a
    self.check_tracker(ic, 'bar', default_value, self.a, 1)
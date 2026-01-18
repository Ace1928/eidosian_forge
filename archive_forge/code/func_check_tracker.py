import unittest
import warnings
from traits.api import (
def check_tracker(self, object, trait, old, new, count):
    self.assertEqual(count, self.changed_count)
    self.assertIs(object, self.changed_object)
    self.assertEqual(trait, self.changed_trait)
    self.assertIs(old, self.changed_old)
    self.assertIs(new, self.changed_new)
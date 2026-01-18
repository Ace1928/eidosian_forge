import unittest
import warnings
from traits.api import (
def reset_change_tracker(self):
    self.changed_object = None
    self.changed_trait = None
    self.changed_old = None
    self.changed_new = None
    self.changed_count = 0
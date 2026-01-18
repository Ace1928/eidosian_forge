import unittest
import warnings
from traits.api import (
from traits.testing.optional_dependencies import requires_traitsui
def assertLastTraitListEventEqual(self, index, removed, added):
    self.assertEqual(self.last_event.index, index)
    self.assertEqual(self.last_event.removed, removed)
    self.assertEqual(self.last_event.added, added)
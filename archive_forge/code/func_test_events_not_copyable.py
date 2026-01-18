import unittest
from traits.api import (
def test_events_not_copyable(self):
    self.assertNotIn('e', self.names)
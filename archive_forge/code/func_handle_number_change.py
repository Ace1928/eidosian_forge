import unittest
from traits.api import (
from traits.observation.api import (
@observe('number')
def handle_number_change(self, event):
    self.number_change_events.append(event)
import unittest
from traits.api import (
from traits.observation.api import (
@observe(trait('records').list_items().trait('number'))
def handle_record_number_changed(self, event):
    self.record_number_change_events.append(event)
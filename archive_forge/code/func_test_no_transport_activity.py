import unittest
from ... import tests, transport, ui
from ..ui_testing import StringIOAsTTY, StringIOWithEncoding, TextUIFactory
def test_no_transport_activity(self):
    t = transport.get_transport_from_url('memory:///')
    self.factory.log_transport_activity(display=True)
    self._check_log_transport_activity_display_no_bytes()
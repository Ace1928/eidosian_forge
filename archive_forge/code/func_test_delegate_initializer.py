import unittest
from traits.api import (
from traits.observation.api import (
def test_delegate_initializer(self):
    mess = DelegateMess()
    self.assertFalse(mess.handler_called)
    mess.dummy1.x = 20
    self.assertTrue(mess.handler_called)
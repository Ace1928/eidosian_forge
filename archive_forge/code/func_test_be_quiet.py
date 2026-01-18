import unittest
from ... import tests, transport, ui
from ..ui_testing import StringIOAsTTY, StringIOWithEncoding, TextUIFactory
def test_be_quiet(self):
    self.factory.be_quiet(True)
    self.assertEqual(True, self.factory.is_quiet())
    self.factory.be_quiet(False)
    self.assertEqual(False, self.factory.is_quiet())
import unittest
from ... import tests, transport, ui
from ..ui_testing import StringIOAsTTY, StringIOWithEncoding, TextUIFactory
def test_confirm_action(self):
    self._load_responses([True])
    result = self.factory.confirm_action('Break a lock?', 'bzr.lock.break.confirm', {})
    self.assertEqual(result, True)
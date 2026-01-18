from unittest import TestCase
from fastimport import (
def test_InvalidTimezone(self):
    e = errors.InvalidTimezone(99, 'aa:bb')
    self.assertEqual('aa:bb', e.timezone)
    self.assertEqual('', e.reason)
    self.assertEqual("line 99: Timezone 'aa:bb' could not be converted.", str(e))
    e = errors.InvalidTimezone(99, 'aa:bb', 'Non-numeric hours')
    self.assertEqual('aa:bb', e.timezone)
    self.assertEqual(' Non-numeric hours', e.reason)
    self.assertEqual("line 99: Timezone 'aa:bb' could not be converted. Non-numeric hours", str(e))
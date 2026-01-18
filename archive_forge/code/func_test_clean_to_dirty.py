import breezy.errors as errors
import breezy.transactions as transactions
from breezy.tests import TestCase
def test_clean_to_dirty(self):
    weave = DummyWeave('A weave')
    self.transaction.map.add_weave('id', weave)
    self.transaction.register_clean(weave)
    self.transaction.register_dirty(weave)
    self.assertTrue(self.transaction.is_dirty(weave))
    self.assertFalse(self.transaction.is_clean(weave))
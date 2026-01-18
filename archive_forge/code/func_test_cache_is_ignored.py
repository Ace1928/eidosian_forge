import breezy.errors as errors
import breezy.transactions as transactions
from breezy.tests import TestCase
def test_cache_is_ignored(self):
    transaction = transactions.PassThroughTransaction()
    transaction.set_cache_size(100)
    weave = 'a weave'
    transaction.map.add_weave('id', weave)
    self.assertEqual(None, transaction.map.find_weave('id'))
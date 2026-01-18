import breezy.errors as errors
import breezy.transactions as transactions
from breezy.tests import TestCase
def test_add_and_get(self):
    weave = 'a weave'
    self.transaction.map.add_weave('id', weave)
    self.assertEqual(weave, self.transaction.map.find_weave('id'))
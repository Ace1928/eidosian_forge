import breezy.errors as errors
import breezy.transactions as transactions
from breezy.tests import TestCase
def test_finish_tells_versioned_file_finished(self):
    weave = DummyWeave('a weave')
    self.transaction.register_dirty(weave)
    self.transaction.finish()
    self.assertTrue(weave.finished)
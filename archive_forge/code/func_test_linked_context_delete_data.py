import testtools
from testtools import matchers
from yaql.language import contexts
from yaql.language import specs
def test_linked_context_delete_data(self):
    mc = self.create_linked_context()
    self.assertIn('key', mc)
    del mc['key']
    self.assertNotIn('key', mc)
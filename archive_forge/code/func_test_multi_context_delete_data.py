import testtools
from testtools import matchers
from yaql.language import contexts
from yaql.language import specs
def test_multi_context_delete_data(self):
    mc = self.create_multi_context()
    del mc['key']
    self.assertNotIn('key', mc)
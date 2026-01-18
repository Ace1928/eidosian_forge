import testtools
from testtools import matchers
from yaql.language import contexts
from yaql.language import specs
def test_multi_context_data_in(self):
    mc = self.create_multi_context()
    self.assertIn('key', mc)
    self.assertIn('key4', mc)
    self.assertNotIn('key2', mc)
    self.assertIn('key2', mc.parent)
    self.assertNotIn('key3', mc.parent)
    self.assertNotIn('key4', mc.parent)
    self.assertIn('key3', mc.parent.parent)
    self.assertIsNone(mc.parent.parent.parent)
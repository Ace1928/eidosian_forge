import testtools
from testtools import matchers
from yaql.language import contexts
from yaql.language import specs
def test_multi_context_data(self):
    mc = self.create_multi_context()
    self.assertEqual(mc['key'], 'context3')
    self.assertEqual(mc['key2'], 'context4')
    self.assertEqual(mc['key3'], 'context1')
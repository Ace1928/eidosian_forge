import testtools
from testtools import matchers
from yaql.language import contexts
from yaql.language import specs
def test_linked_context_data(self):
    mc = self.create_linked_context()
    self.assertEqual(mc['key'], 'context3')
    self.assertEqual(mc['key2'], 'context2')
    self.assertEqual(mc['key3'], 'context3')
    self.assertEqual(mc['key3'], 'context3')
    self.assertEqual(mc.parent['key'], 'context2')
    self.assertEqual(mc.parent.parent['key'], 'context1')
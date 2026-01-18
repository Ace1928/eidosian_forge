from ... import errors, tests, transport
from .. import index as _mod_index
def test_key_count_buffers(self):
    index = self.make_index(nodes=self.make_nodes(2))
    del index._transport._activity[:]
    self.assertEqual(2, index.key_count())
    self.assertEqual([('readv', 'index', [(0, 200)], True, index._size)], index._transport._activity)
    self.assertIsNot(None, index._nodes)
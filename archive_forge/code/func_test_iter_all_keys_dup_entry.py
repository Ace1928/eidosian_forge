from ... import errors, tests, transport
from .. import index as _mod_index
def test_iter_all_keys_dup_entry(self):
    idx1 = self.make_index('1', 1, nodes=[((b'name',), b'data', ([(b'ref',)],)), ((b'ref',), b'refdata', ([],))])
    idx2 = self.make_index('2', 1, nodes=[((b'ref',), b'refdata', ([],))])
    idx = _mod_index.CombinedGraphIndex([idx1, idx2])
    self.assertEqual({(idx1, (b'name',), b'data', (((b'ref',),),)), (idx1, (b'ref',), b'refdata', ((),))}, set(idx.iter_entries([(b'name',), (b'ref',)])))
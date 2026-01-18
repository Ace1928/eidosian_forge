from ... import errors, tests, transport
from .. import index as _mod_index
def test_iter_all_entries_references(self):
    index = self.make_index(1, nodes=[((b'name',), b'data', ([(b'ref',)],)), ((b'ref',), b'refdata', ([],))])
    self.assertEqual({(index, (b'name',), b'data', (((b'ref',),),)), (index, (b'ref',), b'refdata', ((),))}, set(index.iter_all_entries()))
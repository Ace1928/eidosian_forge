from ... import errors, tests, transport
from .. import index as _mod_index
def test_construct_with_callback(self):
    idx = _mod_index.InMemoryGraphIndex()
    adapter = _mod_index.GraphIndexPrefixAdapter(idx, (b'prefix',), 1, idx.add_nodes)
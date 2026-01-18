from ... import errors, tests, transport
from .. import index as _mod_index
def test_add_node(self):
    index, adapter = self.make_index(add_callback=True)
    adapter.add_node((b'key',), b'value', (((b'ref',),),))
    self.assertEqual({(index, (b'prefix', b'key'), b'value', (((b'prefix', b'ref'),),))}, set(index.iter_all_entries()))
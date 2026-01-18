from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def assertIterInteresting(self, records, items, interesting_keys, old_keys):
    """Check the result of iter_interesting_nodes.

        Note that we no longer care how many steps are taken, etc, just that
        the right contents are returned.

        :param records: A list of record keys that should be yielded
        :param items: A list of items (key,value) that should be yielded.
        """
    store = self.get_chk_bytes()
    store._search_key_func = chk_map._search_key_plain
    iter_nodes = chk_map.iter_interesting_nodes(store, interesting_keys, old_keys)
    record_keys = []
    all_items = []
    for record, new_items in iter_nodes:
        if record is not None:
            record_keys.append(record.key)
        if new_items:
            all_items.extend(new_items)
    self.assertEqual(sorted(records), sorted(record_keys))
    self.assertEqual(sorted(items), sorted(all_items))
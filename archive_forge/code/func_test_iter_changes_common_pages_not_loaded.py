from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_iter_changes_common_pages_not_loaded(self):
    basis_dict = {(b'aaa',): b'foo bar', (b'aab',): b'common altered a', (b'b',): b'foo bar b'}
    target_dict = {(b'aaa',): b'foo bar', (b'aab',): b'common altered b', (b'at',): b'foo bar t'}
    basis = self._get_map(basis_dict, maximum_size=10)
    target = self._get_map(target_dict, maximum_size=10, chk_bytes=basis._store)
    basis_get = basis._store.get_record_stream

    def get_record_stream(keys, order, fulltext):
        if (b'sha1:1adf7c0d1b9140ab5f33bb64c6275fa78b1580b7',) in keys:
            raise AssertionError("'aaa' pointer was followed %r" % keys)
        return basis_get(keys, order, fulltext)
    basis._store.get_record_stream = get_record_stream
    result = sorted(list(target.iter_changes(basis)))
    for change in result:
        if change[0] == (b'aaa',):
            self.fail('Found unexpected change: %s' % change)
import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
def test_empty_2_1(self):
    builder = btree_index.BTreeBuilder(key_elements=2, reference_lists=1)
    temp_file = builder.finish()
    content = temp_file.read()
    del temp_file
    self.assertEqual(b'B+Tree Graph Index 2\nnode_ref_lists=1\nkey_elements=2\nlen=0\nrow_lengths=\n', content)
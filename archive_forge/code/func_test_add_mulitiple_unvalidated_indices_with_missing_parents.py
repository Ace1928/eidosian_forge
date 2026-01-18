import gzip
import sys
from io import BytesIO
from patiencediff import PatienceSequenceMatcher
from ... import errors, multiparent, osutils, tests
from ... import transport as _mod_transport
from ...tests import (TestCase, TestCaseWithMemoryTransport,
from .. import knit, knitpack_repo, pack, pack_repo
from ..index import *
from ..knit import (AnnotatedKnitContent, KnitContent, KnitCorrupt,
from ..versionedfile import (AbsentContentFactory, ConstantMapper,
def test_add_mulitiple_unvalidated_indices_with_missing_parents(self):
    g_index_1 = self.make_new_missing_parent_g_index('one')
    g_index_2 = self.make_new_missing_parent_g_index('two')
    combined = CombinedGraphIndex([g_index_1, g_index_2])
    index = _KnitGraphIndex(combined, lambda: True, deltas=True)
    index.scan_unvalidated_index(g_index_1)
    index.scan_unvalidated_index(g_index_2)
    self.assertEqual(frozenset([(b'one-missing-parent',), (b'two-missing-parent',)]), index.get_missing_compression_parents())
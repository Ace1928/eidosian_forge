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
def test_add_mulitiple_unvalidated_indices_with_mutual_dependencies(self):
    graph_index_a = self.make_g_index('one', 2, [((b'parent-one',), b' 100 78', ([(b'non-compression-parent',)], [])), ((b'child-of-two',), b' 100 78', ([(b'parent-two',)], [(b'parent-two',)]))])
    graph_index_b = self.make_g_index('two', 2, [((b'parent-two',), b' 100 78', ([(b'non-compression-parent',)], [])), ((b'child-of-one',), b' 100 78', ([(b'parent-one',)], [(b'parent-one',)]))])
    combined = CombinedGraphIndex([graph_index_a, graph_index_b])
    index = _KnitGraphIndex(combined, lambda: True, deltas=True)
    index.scan_unvalidated_index(graph_index_a)
    index.scan_unvalidated_index(graph_index_b)
    self.assertEqual(frozenset([]), index.get_missing_compression_parents())
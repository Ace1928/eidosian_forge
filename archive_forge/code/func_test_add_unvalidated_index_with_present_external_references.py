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
def test_add_unvalidated_index_with_present_external_references(self):
    index = self.two_graph_index(deltas=True)
    unvalidated = index._graph_index._indices[1]
    index.scan_unvalidated_index(unvalidated)
    self.assertEqual(frozenset(), index.get_missing_compression_parents())
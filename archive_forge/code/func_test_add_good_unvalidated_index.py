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
def test_add_good_unvalidated_index(self):
    unvalidated = self.make_g_index('unvalidated')
    combined = CombinedGraphIndex([unvalidated])
    index = _KnitGraphIndex(combined, lambda: True, parents=False)
    index.scan_unvalidated_index(unvalidated)
    self.assertEqual(frozenset(), index.get_missing_compression_parents())
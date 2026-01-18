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
def test_add_versions_deltas(self):
    index = self.two_graph_index(deltas=True, catch_adds=True)
    index.add_records([((b'new',), b'fulltext,no-eol', (None, 50, 60), [(b'separate',)]), ((b'new2',), b'line-delta', (None, 0, 6), [(b'new',)])])
    self.assertEqual([((b'new',), b'N50 60', (((b'separate',),), ())), ((b'new2',), b' 0 6', (((b'new',),), ((b'new',),)))], sorted(self.caught_entries[0]))
    self.assertEqual(1, len(self.caught_entries))
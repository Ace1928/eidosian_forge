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
def test__expand_delta_comp_parent_not_available(self):
    ann = self.make_annotator()
    rev_key = (b'rev-id',)
    parent_key = (b'parent-id',)
    record = [b'0,1,1\n', b'new-line\n']
    details = ('line-delta', False)
    res = ann._expand_record(rev_key, (parent_key,), parent_key, record, details)
    self.assertEqual(None, res)
    self.assertTrue(parent_key in ann._pending_deltas)
    pending = ann._pending_deltas[parent_key]
    self.assertEqual(1, len(pending))
    self.assertEqual((rev_key, (parent_key,), record, details), pending[0])
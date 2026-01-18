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
def test__process_pending(self):
    ann = self.make_annotator()
    rev_key = (b'rev-id',)
    p1_key = (b'p1-id',)
    p2_key = (b'p2-id',)
    record = [b'0,1,1\n', b'new-line\n']
    details = ('line-delta', False)
    p1_record = [b'line1\n', b'line2\n']
    ann._num_compression_children[p1_key] = 1
    res = ann._expand_record(rev_key, (p1_key, p2_key), p1_key, record, details)
    self.assertEqual(None, res)
    self.assertEqual({}, ann._pending_annotation)
    res = ann._expand_record(p1_key, (), None, p1_record, ('fulltext', False))
    self.assertEqual(p1_record, res)
    ann._annotations_cache[p1_key] = [(p1_key,)] * 2
    res = ann._process_pending(p1_key)
    self.assertEqual([], res)
    self.assertFalse(p1_key in ann._pending_deltas)
    self.assertTrue(p2_key in ann._pending_annotation)
    self.assertEqual({p2_key: [(rev_key, (p1_key, p2_key))]}, ann._pending_annotation)
    res = ann._expand_record(p2_key, (), None, [], ('fulltext', False))
    ann._annotations_cache[p2_key] = []
    res = ann._process_pending(p2_key)
    self.assertEqual([rev_key], res)
    self.assertEqual({}, ann._pending_annotation)
    self.assertEqual({}, ann._pending_deltas)
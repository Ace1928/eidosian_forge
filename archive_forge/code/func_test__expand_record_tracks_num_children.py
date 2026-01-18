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
def test__expand_record_tracks_num_children(self):
    ann = self.make_annotator()
    rev_key = (b'rev-id',)
    rev2_key = (b'rev2-id',)
    parent_key = (b'parent-id',)
    record = [b'0,1,1\n', b'new-line\n']
    details = ('line-delta', False)
    ann._num_compression_children[parent_key] = 2
    ann._expand_record(parent_key, (), None, [b'line1\n', b'line2\n'], ('fulltext', False))
    res = ann._expand_record(rev_key, (parent_key,), parent_key, record, details)
    self.assertEqual({parent_key: 1}, ann._num_compression_children)
    res = ann._expand_record(rev2_key, (parent_key,), parent_key, record, details)
    self.assertFalse(parent_key in ann._content_objects)
    self.assertEqual({}, ann._num_compression_children)
    self.assertEqual({}, ann._content_objects)
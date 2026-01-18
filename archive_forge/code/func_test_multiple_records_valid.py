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
def test_multiple_records_valid(self):
    total_txt, record_1, record_2 = self.make_multiple_records()
    transport = MockTransport([b''.join(total_txt)])
    access = _KnitKeyAccess(transport, ConstantMapper('filename'))
    knit = KnitVersionedFiles(None, access)
    records = [((b'rev-id-1',), ((b'rev-id-1',), record_1[0], record_1[1])), ((b'rev-id-2',), ((b'rev-id-2',), record_2[0], record_2[1]))]
    contents = list(knit._read_records_iter(records))
    self.assertEqual([((b'rev-id-1',), [b'foo\n', b'bar\n'], record_1[2]), ((b'rev-id-2',), [b'baz\n'], record_2[2])], contents)
    raw_contents = list(knit._read_records_iter_raw(records))
    self.assertEqual([((b'rev-id-1',), total_txt[0], record_1[2]), ((b'rev-id-2',), total_txt[1], record_2[2])], raw_contents)
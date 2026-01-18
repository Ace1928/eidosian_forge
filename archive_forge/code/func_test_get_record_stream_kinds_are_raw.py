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
def test_get_record_stream_kinds_are_raw(self):
    vf = self.make_test_knit(name='test')
    vf.add_lines((b'base',), (), [b'base\n', b'content\n'])
    vf.add_lines((b'd1',), ((b'base',),), [b'd1\n'])
    vf.add_lines((b'd2',), ((b'd1',),), [b'd2\n'])
    keys = [(b'base',), (b'd1',), (b'd2',)]
    generator = _VFContentMapGenerator(vf, keys, global_map=vf.get_parent_map(keys))
    kinds = {(b'base',): 'knit-delta-closure', (b'd1',): 'knit-delta-closure-ref', (b'd2',): 'knit-delta-closure-ref'}
    for record in generator.get_record_stream():
        self.assertEqual(kinds[record.key], record.storage_kind)
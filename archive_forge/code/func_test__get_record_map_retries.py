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
def test__get_record_map_retries(self):
    vf, reload_counter = self.make_vf_for_retrying()
    keys = [(b'rev-1',), (b'rev-2',), (b'rev-3',)]
    records = vf._get_record_map(keys)
    self.assertEqual(keys, sorted(records.keys()))
    self.assertEqual([1, 1, 0], reload_counter)
    for trans, name in vf._access._indices.values():
        trans.delete(name)
    self.assertRaises(_mod_transport.NoSuchFile, vf._get_record_map, keys)
    self.assertEqual([2, 1, 1], reload_counter)
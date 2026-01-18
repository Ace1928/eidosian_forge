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
def test_failing_readv_raises_retry(self):
    memos = self.make_pack_file()
    transport = self.get_transport()
    failing_transport = MockReadvFailingTransport([transport.get_bytes('packname')])
    reload_called, reload_func = self.make_reload_func()
    access = pack_repo._DirectPackAccess({'foo': (failing_transport, 'packname')}, reload_func=reload_func)
    self.assertEqual([b'1234567890'], list(access.get_raw_records(memos[:1])))
    self.assertEqual([b'12345'], list(access.get_raw_records(memos[1:2])))
    e = self.assertListRaises(pack_repo.RetryWithNewPacks, access.get_raw_records, memos)
    self.assertFalse(e.reload_occurred)
    self.assertIsInstance(e.exc_info, tuple)
    self.assertIs(e.exc_info[0], _mod_transport.NoSuchFile)
    self.assertIsInstance(e.exc_info[1], _mod_transport.NoSuchFile)
    self.assertEqual('packname', e.exc_info[1].path)
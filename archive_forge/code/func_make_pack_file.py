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
def make_pack_file(self):
    """Create a pack file with 2 records."""
    access, writer = self._get_access(packname='packname', index='foo')
    memos = []
    memos.extend(access.add_raw_records([(b'key1', 10)], [b'1234567890']))
    memos.extend(access.add_raw_records([(b'key2', 5)], [b'12345']))
    writer.end()
    return memos
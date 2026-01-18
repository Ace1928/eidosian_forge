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
def test__get_total_build_size(self):
    positions = {(b'a',): (('fulltext', False), ((b'a',), 0, 100), None), (b'b',): (('line-delta', False), ((b'b',), 100, 21), (b'a',)), (b'c',): (('line-delta', False), ((b'c',), 121, 35), (b'b',)), (b'd',): (('line-delta', False), ((b'd',), 156, 12), (b'b',))}
    self.assertTotalBuildSize(100, [(b'a',)], positions)
    self.assertTotalBuildSize(121, [(b'b',)], positions)
    self.assertTotalBuildSize(156, [(b'c',)], positions)
    self.assertTotalBuildSize(156, [(b'b',), (b'c',)], positions)
    self.assertTotalBuildSize(133, [(b'd',)], positions)
    self.assertTotalBuildSize(168, [(b'c',), (b'd',)], positions)
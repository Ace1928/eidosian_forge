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
def assertA1Only():
    self.assertEqual({(b'a-1',)}, set(idx.keys()))
    self.assertEqual({(b'a-1',): (((b'a-1',), 0, 0), None, (), ('fulltext', False))}, idx.get_build_details([(b'a-1',)]))
    self.assertEqual({(b'a-1',): ()}, idx.get_parent_map(idx.keys()))
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
def test_get_options_deltas(self):
    index = self.two_graph_index(deltas=True)
    self.assertEqual([b'fulltext', b'no-eol'], index.get_options((b'tip',)))
    self.assertEqual([b'line-delta'], index.get_options((b'parent',)))
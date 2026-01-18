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
def test_line_delta(self):
    content1 = self._make_content([('', 'a'), ('', 'b')])
    content2 = self._make_content([('', 'a'), ('', 'a'), ('', 'c')])
    self.assertEqual(content1.line_delta(content2), [(1, 2, 2, [('', 'a'), ('', 'c')])])
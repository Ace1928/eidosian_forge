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
def test_knit_index_checks_header(self):
    t = _mod_transport.get_transport_from_path('.')
    t.put_bytes('test.kndx', b'# not really a knit header\n\n')
    k = self.make_test_knit()
    self.assertRaises(KnitHeaderError, k.keys)
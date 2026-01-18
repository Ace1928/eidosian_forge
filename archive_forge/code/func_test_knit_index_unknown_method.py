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
def test_knit_index_unknown_method(self):
    error = KnitIndexUnknownMethod('http://host/foo.kndx', ['bad', 'no-eol'])
    self.assertEqual("Knit index http://host/foo.kndx does not have a known method in options: ['bad', 'no-eol']", str(error))
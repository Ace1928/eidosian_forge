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
def test_delay_create_and_add_versions(self):
    transport = MockTransport()
    index = self.get_knit_index(transport, 'filename', 'w')
    self.assertEqual([], transport.calls)
    self.add_a_b(index)
    self.assertEqual(2, len(transport.calls))
    call = transport.calls.pop(0)
    self.assertEqual('put_file_non_atomic', call[0])
    self.assertEqual('filename.kndx', call[1][0])
    self.assertEqual(_KndxIndex.HEADER, call[1][1].getvalue())
    self.assertEqual({'create_parent_dir': True}, call[2])
    call = transport.calls.pop(0)
    self.assertEqual('put_file_non_atomic', call[0])
    self.assertEqual('filename.kndx', call[1][0])
    self.assertEqual(_KndxIndex.HEADER + b'\na option 0 1 .b :\na opt 1 2 .c :\nb option 2 3 0 :', call[1][1].getvalue())
    self.assertEqual({'create_parent_dir': True}, call[2])
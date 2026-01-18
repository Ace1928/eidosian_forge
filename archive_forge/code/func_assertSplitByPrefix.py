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
def assertSplitByPrefix(self, expected_map, expected_prefix_order, keys):
    split, prefix_order = KnitVersionedFiles._split_by_prefix(keys)
    self.assertEqual(expected_map, split)
    self.assertEqual(expected_prefix_order, prefix_order)
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
def assertGroupKeysForIo(self, exp_groups, keys, non_local_keys, positions, _min_buffer_size=None):
    kvf = self.make_test_knit()
    if _min_buffer_size is None:
        _min_buffer_size = knit._STREAM_MIN_BUFFER_SIZE
    self.assertEqual(exp_groups, kvf._group_keys_for_io(keys, non_local_keys, positions, _min_buffer_size=_min_buffer_size))
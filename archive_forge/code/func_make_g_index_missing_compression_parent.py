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
def make_g_index_missing_compression_parent(self):
    graph_index = self.make_g_index('missing_comp', 2, [((b'tip',), b' 100 78', ([(b'missing-parent',), (b'ghost',)], [(b'missing-parent',)]))])
    return graph_index
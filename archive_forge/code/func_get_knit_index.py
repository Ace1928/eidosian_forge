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
def get_knit_index(self, transport, name, mode):
    mapper = ConstantMapper(name)
    self.overrideAttr(knit, '_load_data', self._load_data)

    def allow_writes():
        return 'w' in mode
    return _KndxIndex(transport, mapper, lambda: None, allow_writes, lambda: True)
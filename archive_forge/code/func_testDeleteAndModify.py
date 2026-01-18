import itertools
from gzip import GzipFile
from io import BytesIO
from ... import errors
from ... import graph as _mod_graph
from ... import osutils, progress, transport, ui
from ...errors import RevisionAlreadyPresent, RevisionNotPresent
from ...tests import (TestCase, TestCaseWithMemoryTransport, TestNotApplicable,
from ...tests.http_utils import TestCaseWithWebserver
from ...tests.scenarios import load_tests_apply_scenarios
from ...transport.memory import MemoryTransport
from .. import groupcompress
from .. import knit as _mod_knit
from .. import versionedfile as versionedfile
from ..knit import cleanup_pack_knit, make_file_factory, make_pack_factory
from ..versionedfile import (ChunkedContentFactory, ConstantMapper,
from ..weave import WeaveFile, WeaveInvalidChecksum
from ..weavefile import write_weave
def testDeleteAndModify(self):
    """Clashing delete and modification.

        If one side modifies a region and the other deletes it then
        there should be a conflict with one side blank.
        """
    return
    self.doMerge([b'aaa', b'bbb', b'ccc'], [b'aaa', b'ddd', b'ccc'], [b'aaa', b'ccc'], [b'<<<<<<<< ', b'aaa', b'=======', b'>>>>>>> ', b'ccc'])
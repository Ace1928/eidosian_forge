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
def test_deletion_extended(self):
    """One side deletes, the other deletes more.
        """
    base = b'            line 1\n            line 2\n            line 3\n            '
    a = b'            line 1\n            line 2\n            '
    b = b'            line 1\n            '
    result = b'            line 1\n<<<<<<< \n            line 2\n=======\n>>>>>>> \n            '
    self._test_merge_from_strings(base, a, b, result)
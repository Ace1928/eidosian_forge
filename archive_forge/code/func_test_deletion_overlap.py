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
def test_deletion_overlap(self):
    """Delete overlapping regions with no other conflict.

        Arguably it'd be better to treat these as agreement, rather than
        conflict, but for now conflict is safer.
        """
    base = b'            start context\n            int a() {}\n            int b() {}\n            int c() {}\n            end context\n            '
    a = b'            start context\n            int a() {}\n            end context\n            '
    b = b'            start context\n            int c() {}\n            end context\n            '
    result = b'            start context\n<<<<<<< \n            int a() {}\n=======\n            int c() {}\n>>>>>>> \n            end context\n            '
    self._test_merge_from_strings(base, a, b, result)
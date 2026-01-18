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
def test_iter_lines_added_or_present_in_keys(self):
    self._lines[b'A'] = [b'FOO', b'BAR']
    self._lines[b'B'] = [b'HEY']
    self._lines[b'C'] = [b'Alberta']
    it = self.texts.iter_lines_added_or_present_in_keys([(b'A',), (b'B',)])
    self.assertEqual(sorted([(b'FOO', b'A'), (b'BAR', b'A'), (b'HEY', b'B')]), sorted(list(it)))
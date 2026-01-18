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
def test_empty_lines(self):
    """Empty files can be stored."""
    f = self.get_versionedfiles()
    key_a = self.get_simple_key(b'a')
    f.add_lines(key_a, [], [])
    self.assertEqual(b'', next(f.get_record_stream([key_a], 'unordered', True)).get_bytes_as('fulltext'))
    key_b = self.get_simple_key(b'b')
    f.add_lines(key_b, self.get_parents([key_a]), [])
    self.assertEqual(b'', next(f.get_record_stream([key_b], 'unordered', True)).get_bytes_as('fulltext'))
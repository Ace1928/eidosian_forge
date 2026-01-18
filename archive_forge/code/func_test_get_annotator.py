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
def test_get_annotator(self):
    files = self.get_versionedfiles()
    self.get_diamond_files(files)
    origin_key = self.get_simple_key(b'origin')
    base_key = self.get_simple_key(b'base')
    left_key = self.get_simple_key(b'left')
    right_key = self.get_simple_key(b'right')
    merged_key = self.get_simple_key(b'merged')
    origins, lines = files.get_annotator().annotate(origin_key)
    self.assertEqual([(origin_key,)], origins)
    self.assertEqual([b'origin\n'], lines)
    origins, lines = files.get_annotator().annotate(base_key)
    self.assertEqual([(base_key,)], origins)
    origins, lines = files.get_annotator().annotate(merged_key)
    if self.graph:
        self.assertEqual([(base_key,), (left_key,), (right_key,), (merged_key,)], origins)
    else:
        self.assertEqual([(merged_key,), (merged_key,), (merged_key,), (merged_key,)], origins)
    self.assertRaises(RevisionNotPresent, files.get_annotator().annotate, self.get_simple_key(b'missing-key'))
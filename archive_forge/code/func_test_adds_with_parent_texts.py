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
def test_adds_with_parent_texts(self):
    f = self.get_file()
    parent_texts = {}
    _, _, parent_texts[b'r0'] = f.add_lines(b'r0', [], [b'a\n', b'b\n'])
    try:
        _, _, parent_texts[b'r1'] = f.add_lines_with_ghosts(b'r1', [b'r0', b'ghost'], [b'b\n', b'c\n'], parent_texts=parent_texts)
    except NotImplementedError:
        _, _, parent_texts[b'r1'] = f.add_lines(b'r1', [b'r0'], [b'b\n', b'c\n'], parent_texts=parent_texts)
    f.add_lines(b'r2', [b'r1'], [b'c\n', b'd\n'], parent_texts=parent_texts)
    self.assertNotEqual(None, parent_texts[b'r0'])
    self.assertNotEqual(None, parent_texts[b'r1'])

    def verify_file(f):
        versions = f.versions()
        self.assertTrue(b'r0' in versions)
        self.assertTrue(b'r1' in versions)
        self.assertTrue(b'r2' in versions)
        self.assertEqual(f.get_lines(b'r0'), [b'a\n', b'b\n'])
        self.assertEqual(f.get_lines(b'r1'), [b'b\n', b'c\n'])
        self.assertEqual(f.get_lines(b'r2'), [b'c\n', b'd\n'])
        self.assertEqual(3, f.num_versions())
        origins = f.annotate(b'r1')
        self.assertEqual(origins[0][0], b'r0')
        self.assertEqual(origins[1][0], b'r1')
        origins = f.annotate(b'r2')
        self.assertEqual(origins[0][0], b'r1')
        self.assertEqual(origins[1][0], b'r2')
    verify_file(f)
    f = self.reopen_file()
    verify_file(f)
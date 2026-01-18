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
def test_add_lines_with_ghosts_after_normal_revs(self):
    vf = self.get_file()
    try:
        vf.add_lines_with_ghosts(b'base', [], [b'line\n', b'line_b\n'])
    except NotImplementedError:
        return
    vf.add_lines_with_ghosts(b'references_ghost', [b'base', b'a_ghost'], [b'line\n', b'line_b\n', b'line_c\n'])
    origins = vf.annotate(b'references_ghost')
    self.assertEqual((b'base', b'line\n'), origins[0])
    self.assertEqual((b'base', b'line_b\n'), origins[1])
    self.assertEqual((b'references_ghost', b'line_c\n'), origins[2])
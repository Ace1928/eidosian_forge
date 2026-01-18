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
def test_add_unchanged_last_line_noeol_snapshot(self):
    """Add a text with an unchanged last line with no eol should work."""
    for length in range(20):
        version_lines = {}
        vf = self.get_file('case-%d' % length)
        prefix = b'step-%d'
        parents = []
        for step in range(length):
            version = prefix % step
            lines = [b'prelude \n'] * step + [b'line']
            vf.add_lines(version, parents, lines)
            version_lines[version] = lines
            parents = [version]
        vf.add_lines(b'no-eol', parents, [b'line'])
        vf.get_texts(version_lines.keys())
        self.assertEqualDiff(b'line', vf.get_text(b'no-eol'))
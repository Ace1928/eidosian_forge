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
def test_add_lines_with_ghosts_nostoresha(self):
    """When nostore_sha is supplied using old content raises."""
    vf = self.get_file()
    empty_text = (b'a', [])
    sample_text_nl = (b'b', [b'foo\n', b'bar\n'])
    sample_text_no_nl = (b'c', [b'foo\n', b'bar'])
    shas = []
    for version, lines in (empty_text, sample_text_nl, sample_text_no_nl):
        sha, _, _ = vf.add_lines(version, [], lines)
        shas.append(sha)
    try:
        vf.add_lines_with_ghosts(b'd', [], [])
    except NotImplementedError:
        raise TestSkipped('add_lines_with_ghosts is optional')
    for sha, (version, lines) in zip(shas, (empty_text, sample_text_nl, sample_text_no_nl)):
        self.assertRaises(ExistingContent, vf.add_lines_with_ghosts, version + b'2', [], lines, nostore_sha=sha)
        self.assertRaises(errors.RevisionNotPresent, vf.get_lines, version + b'2')
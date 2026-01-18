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
def test_add_unicode_content(self):
    vf = self.get_file()
    self.assertRaises(errors.BzrBadParameterUnicode, vf.add_lines, b'a', [], [b'a\n', 'b\n', b'c\n'])
    self.assertRaises((errors.BzrBadParameterUnicode, NotImplementedError), vf.add_lines_with_ghosts, b'a', [], [b'a\n', 'b\n', b'c\n'])
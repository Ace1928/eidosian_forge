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
def test_add_fallback_implies_without_fallbacks(self):
    f = self.get_versionedfiles('files')
    if getattr(f, 'add_fallback_versioned_files', None) is None:
        raise TestNotApplicable("%s doesn't support fallbacks" % (f.__class__.__name__,))
    g = self.get_versionedfiles('fallback')
    key_a = self.get_simple_key(b'a')
    g.add_lines(key_a, [], [b'\n'])
    f.add_fallback_versioned_files(g)
    self.assertTrue(key_a in f.get_parent_map([key_a]))
    self.assertFalse(key_a in f.without_fallbacks().get_parent_map([key_a]))
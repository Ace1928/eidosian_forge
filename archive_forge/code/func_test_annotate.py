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
def test_annotate(self):
    files = self.get_versionedfiles()
    self.get_diamond_files(files)
    if self.key_length == 1:
        prefix = ()
    else:
        prefix = (b'FileA',)
    origins = files.annotate(prefix + (b'origin',))
    self.assertEqual([(prefix + (b'origin',), b'origin\n')], origins)
    origins = files.annotate(prefix + (b'base',))
    self.assertEqual([(prefix + (b'base',), b'base\n')], origins)
    origins = files.annotate(prefix + (b'merged',))
    if self.graph:
        self.assertEqual([(prefix + (b'base',), b'base\n'), (prefix + (b'left',), b'left\n'), (prefix + (b'right',), b'right\n'), (prefix + (b'merged',), b'merged\n')], origins)
    else:
        self.assertEqual([(prefix + (b'merged',), b'base\n'), (prefix + (b'merged',), b'left\n'), (prefix + (b'merged',), b'right\n'), (prefix + (b'merged',), b'merged\n')], origins)
    self.assertRaises(RevisionNotPresent, files.annotate, prefix + ('missing-key',))
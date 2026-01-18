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
def get_file_corrupted_text(self):
    w = WeaveFile('foo', self.get_transport(), create=True, get_scope=self.get_transaction)
    w.add_lines(b'v1', [], [b'hello\n'])
    w.add_lines(b'v2', [b'v1'], [b'hello\n', b'there\n'])
    self.assertEqual([(b'{', 0), b'hello\n', (b'}', None), (b'{', 1), b'there\n', (b'}', None)], w._weave)
    self.assertEqual([b'f572d396fae9206628714fb2ce00f72e94f2258f', b'90f265c6e75f1c8f9ab76dcf85528352c5f215ef'], w._sha1s)
    w.check()
    w._weave[4] = b'There\n'
    return w
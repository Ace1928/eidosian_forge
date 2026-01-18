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
def test_get_record_stream_ordered(self):
    vf = self.get_ordering_vf({(b'A',): 3, (b'B',): 2, (b'C',): 4, (b'D',): 1})
    request_keys = [(b'B',), (b'C',), (b'D',), (b'A',)]
    keys = [r.key for r in vf.get_record_stream(request_keys, 'unordered', False)]
    self.assertEqual([(b'D',), (b'B',), (b'A',), (b'C',)], keys)
    self.assertEqual([('get_record_stream', request_keys, 'unordered', False)], vf.calls)
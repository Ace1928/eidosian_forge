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
def assertStreamOrder(self, sort_order, seen, keys):
    self.assertEqual(len(set(seen)), len(keys))
    if self.key_length == 1:
        lows = {(): 0}
    else:
        lows = {(b'FileA',): 0, (b'FileB',): 0}
    if not self.graph:
        self.assertEqual(set(keys), set(seen))
    else:
        for key in seen:
            sort_pos = sort_order[key]
            self.assertTrue(sort_pos >= lows[key[:-1]], 'Out of order in sorted stream: {!r}, {!r}'.format(key, seen))
            lows[key[:-1]] = sort_pos
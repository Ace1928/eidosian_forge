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
def test_add_chunks(self):
    f = self.get_versionedfiles()
    key0 = self.get_simple_key(b'r0')
    key1 = self.get_simple_key(b'r1')
    key2 = self.get_simple_key(b'r2')
    keyf = self.get_simple_key(b'foo')

    def add_chunks(key, parents, chunks):
        factory = ChunkedContentFactory(key, parents, osutils.sha_strings(chunks), chunks)
        return f.add_content(factory)
    add_chunks(key0, [], [b'a', b'\nb\n'])
    if self.graph:
        add_chunks(key1, [key0], [b'b', b'\n', b'c\n'])
    else:
        add_chunks(key1, [], [b'b\n', b'c\n'])
    keys = f.keys()
    self.assertIn(key0, keys)
    self.assertIn(key1, keys)
    records = []
    for record in f.get_record_stream([key0, key1], 'unordered', True):
        records.append((record.key, record.get_bytes_as('fulltext')))
    records.sort()
    self.assertEqual([(key0, b'a\nb\n'), (key1, b'b\nc\n')], records)
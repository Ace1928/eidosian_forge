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
def test_get_record_stream_native_formats_are_wire_ready_ft_delta(self):
    files = self.get_versionedfiles()
    target_files = self.get_versionedfiles('target')
    key = self.get_simple_key(b'ft')
    key_delta = self.get_simple_key(b'delta')
    files.add_lines(key, (), [b'my text\n', b'content'])
    if self.graph:
        delta_parents = (key,)
    else:
        delta_parents = ()
    files.add_lines(key_delta, delta_parents, [b'different\n', b'content\n'])
    local = files.get_record_stream([key, key_delta], 'unordered', False)
    ref = files.get_record_stream([key, key_delta], 'unordered', False)
    skipped_records = [0]
    full_texts = {key: b'my text\ncontent', key_delta: b'different\ncontent\n'}
    byte_stream = self.stream_to_bytes_or_skip_counter(skipped_records, full_texts, local)
    network_stream = versionedfile.NetworkRecordStream(byte_stream).read()
    records = []
    target_files.insert_record_stream(self.assertStreamMetaEqual(records, ref, network_stream))
    self.assertEqual(2, len(records) + skipped_records[0])
    if len(records):
        self.assertIdenticalVersionedFile(files, target_files)
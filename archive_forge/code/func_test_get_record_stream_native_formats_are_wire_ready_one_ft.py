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
def test_get_record_stream_native_formats_are_wire_ready_one_ft(self):
    files = self.get_versionedfiles()
    key = self.get_simple_key(b'foo')
    files.add_lines(key, (), [b'my text\n', b'content'])
    stream = files.get_record_stream([key], 'unordered', False)
    record = next(stream)
    if record.storage_kind in ('chunked', 'fulltext'):
        self.assertRecordHasContent(record, b'my text\ncontent')
    else:
        bytes = [record.get_bytes_as(record.storage_kind)]
        network_stream = versionedfile.NetworkRecordStream(bytes).read()
        source_record = record
        records = []
        for record in network_stream:
            records.append(record)
            self.assertEqual(source_record.storage_kind, record.storage_kind)
            self.assertEqual(source_record.parents, record.parents)
            self.assertEqual(source_record.get_bytes_as(source_record.storage_kind), record.get_bytes_as(record.storage_kind))
        self.assertEqual(1, len(records))
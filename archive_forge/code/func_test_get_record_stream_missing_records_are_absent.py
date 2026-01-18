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
def test_get_record_stream_missing_records_are_absent(self):
    files = self.get_versionedfiles()
    self.get_diamond_files(files)
    if self.key_length == 1:
        keys = [(b'merged',), (b'left',), (b'right',), (b'absent',), (b'base',)]
    else:
        keys = [(b'FileA', b'merged'), (b'FileA', b'left'), (b'FileA', b'right'), (b'FileA', b'absent'), (b'FileA', b'base'), (b'FileB', b'merged'), (b'FileB', b'left'), (b'FileB', b'right'), (b'FileB', b'absent'), (b'FileB', b'base'), (b'absent', b'absent')]
    parent_map = files.get_parent_map(keys)
    entries = files.get_record_stream(keys, 'unordered', False)
    self.assertAbsentRecord(files, keys, parent_map, entries)
    entries = files.get_record_stream(keys, 'topological', False)
    self.assertAbsentRecord(files, keys, parent_map, entries)
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
def test_get_record_stream_unknown_storage_kind_raises(self):
    """Asking for a storage kind that the stream cannot supply raises."""
    files = self.get_versionedfiles()
    self.get_diamond_files(files)
    if self.key_length == 1:
        keys = [(b'merged',), (b'left',), (b'right',), (b'base',)]
    else:
        keys = [(b'FileA', b'merged'), (b'FileA', b'left'), (b'FileA', b'right'), (b'FileA', b'base'), (b'FileB', b'merged'), (b'FileB', b'left'), (b'FileB', b'right'), (b'FileB', b'base')]
    parent_map = files.get_parent_map(keys)
    entries = files.get_record_stream(keys, 'unordered', False)
    seen = set()
    for factory in entries:
        seen.add(factory.key)
        self.assertValidStorageKind(factory.storage_kind)
        if factory.sha1 is not None:
            self.assertEqual(files.get_sha1s([factory.key])[factory.key], factory.sha1)
        self.assertEqual(parent_map[factory.key], factory.parents)
        self.assertRaises(UnavailableRepresentation, factory.get_bytes_as, 'mpdiff')
        self.assertIsInstance(factory.get_bytes_as(factory.storage_kind), bytes)
    self.assertEqual(set(keys), seen)
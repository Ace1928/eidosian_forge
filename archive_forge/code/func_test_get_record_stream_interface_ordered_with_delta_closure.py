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
def test_get_record_stream_interface_ordered_with_delta_closure(self):
    """each item must be accessible as a fulltext."""
    files = self.get_versionedfiles()
    self.get_diamond_files(files)
    keys, sort_order = self.get_keys_and_sort_order()
    parent_map = files.get_parent_map(keys)
    entries = files.get_record_stream(keys, 'topological', True)
    seen = []
    for factory in entries:
        seen.append(factory.key)
        self.assertValidStorageKind(factory.storage_kind)
        self.assertSubset([factory.sha1], [None, files.get_sha1s([factory.key])[factory.key]])
        self.assertEqual(parent_map[factory.key], factory.parents)
        ft_bytes = factory.get_bytes_as('fulltext')
        self.assertIsInstance(ft_bytes, bytes)
        chunked_bytes = factory.get_bytes_as('chunked')
        self.assertEqualDiff(ft_bytes, b''.join(chunked_bytes))
        chunked_bytes = factory.iter_bytes_as('chunked')
        self.assertEqualDiff(ft_bytes, b''.join(chunked_bytes))
    self.assertStreamOrder(sort_order, seen, keys)
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
def test_insert_record_stream_delta_missing_basis_no_corruption(self):
    """Insertion where a needed basis is not included notifies the caller
        of the missing basis.  In the meantime a record missing its basis is
        not added.
        """
    source = self.get_knit_delta_source()
    keys = [self.get_simple_key(b'origin'), self.get_simple_key(b'merged')]
    entries = source.get_record_stream(keys, 'unordered', False)
    files = self.get_versionedfiles()
    if self.support_partial_insertion:
        self.assertEqual([], list(files.get_missing_compression_parent_keys()))
        files.insert_record_stream(entries)
        missing_bases = files.get_missing_compression_parent_keys()
        self.assertEqual({self.get_simple_key(b'left')}, set(missing_bases))
        self.assertEqual(set(keys), set(files.get_parent_map(keys)))
    else:
        self.assertRaises(errors.RevisionNotPresent, files.insert_record_stream, entries)
        files.check()
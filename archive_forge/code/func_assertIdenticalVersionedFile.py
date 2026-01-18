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
def assertIdenticalVersionedFile(self, expected, actual):
    """Assert that left and right have the same contents."""
    self.assertEqual(set(actual.keys()), set(expected.keys()))
    actual_parents = actual.get_parent_map(actual.keys())
    if self.graph:
        self.assertEqual(actual_parents, expected.get_parent_map(expected.keys()))
    else:
        for key, parents in actual_parents.items():
            self.assertEqual(None, parents)
    for key in actual.keys():
        actual_text = next(actual.get_record_stream([key], 'unordered', True)).get_bytes_as('fulltext')
        expected_text = next(expected.get_record_stream([key], 'unordered', True)).get_bytes_as('fulltext')
        self.assertEqual(actual_text, expected_text)
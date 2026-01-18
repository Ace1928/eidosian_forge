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
def assertAbsentRecord(self, files, keys, parents, entries):
    """Helper for test_get_record_stream_missing_records_are_absent."""
    seen = set()
    for factory in entries:
        seen.add(factory.key)
        if factory.key[-1] == b'absent':
            self.assertEqual('absent', factory.storage_kind)
            self.assertEqual(None, factory.sha1)
            self.assertEqual(None, factory.parents)
        else:
            self.assertValidStorageKind(factory.storage_kind)
            if factory.sha1 is not None:
                sha1 = files.get_sha1s([factory.key])[factory.key]
                self.assertEqual(sha1, factory.sha1)
            self.assertEqual(parents[factory.key], factory.parents)
            self.assertIsInstance(factory.get_bytes_as(factory.storage_kind), bytes)
    self.assertEqual(set(keys), seen)
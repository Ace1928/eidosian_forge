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
def test_insert_record_stream_existing_keys(self):
    """Inserting keys already in a file should not error."""
    files = self.get_versionedfiles()
    source = self.get_versionedfiles('source')
    self.get_diamond_files(source)
    self.get_diamond_files(files, left_only=True)
    stream = source.get_record_stream(source.keys(), 'topological', False)
    files.insert_record_stream(stream)
    self.assertIdenticalVersionedFile(source, files)
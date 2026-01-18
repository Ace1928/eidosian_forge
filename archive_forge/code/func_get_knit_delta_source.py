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
def get_knit_delta_source(self):
    """Get a source that can produce a stream with knit delta records,
        regardless of this test's scenario.
        """
    mapper = self.get_mapper()
    source_transport = self.get_transport('source')
    source_transport.mkdir('.')
    source = make_file_factory(False, mapper)(source_transport)
    get_diamond_files(source, self.key_length, trailing_eol=True, nograph=False, left_only=False)
    return source
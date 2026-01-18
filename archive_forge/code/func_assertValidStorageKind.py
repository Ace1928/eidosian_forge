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
def assertValidStorageKind(self, storage_kind):
    """Assert that storage_kind is a valid storage_kind."""
    self.assertSubset([storage_kind], ['mpdiff', 'knit-annotated-ft', 'knit-annotated-delta', 'knit-ft', 'knit-delta', 'chunked', 'fulltext', 'knit-annotated-ft-gz', 'knit-annotated-delta-gz', 'knit-ft-gz', 'knit-delta-gz', 'knit-delta-closure', 'knit-delta-closure-ref', 'groupcompress-block', 'groupcompress-block-ref'])
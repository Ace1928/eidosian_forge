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
def test_hash_escaped_mapper(self):
    mapper = versionedfile.HashEscapedPrefixMapper()
    self.assertEqual('88/%2520', mapper.map((b' ', b'revision-id')))
    self.assertEqual('ed/fil%2545-%2549d', mapper.map((b'filE-Id', b'revision-id')))
    self.assertEqual('88/ne%2557-%2549d', mapper.map((b'neW-Id', b'revision-id')))
    self.assertEqual((b'filE-Id',), mapper.unmap('ed/fil%2545-%2549d'))
    self.assertEqual((b'neW-Id',), mapper.unmap('88/ne%2557-%2549d'))
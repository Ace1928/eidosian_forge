import gzip
import sys
from io import BytesIO
from patiencediff import PatienceSequenceMatcher
from ... import errors, multiparent, osutils, tests
from ... import transport as _mod_transport
from ...tests import (TestCase, TestCaseWithMemoryTransport,
from .. import knit, knitpack_repo, pack, pack_repo
from ..index import *
from ..knit import (AnnotatedKnitContent, KnitContent, KnitCorrupt,
from ..versionedfile import (AbsentContentFactory, ConstantMapper,
def assertDerivedBlocksEqual(self, source, target, noeol=False):
    """Assert that the derived matching blocks match real output"""
    source_lines = source.splitlines(True)
    target_lines = target.splitlines(True)

    def nl(line):
        if noeol and (not line.endswith('\n')):
            return line + '\n'
        else:
            return line
    source_content = self._make_content([(None, nl(l)) for l in source_lines])
    target_content = self._make_content([(None, nl(l)) for l in target_lines])
    line_delta = source_content.line_delta(target_content)
    delta_blocks = list(KnitContent.get_line_delta_blocks(line_delta, source_lines, target_lines))
    matcher = PatienceSequenceMatcher(None, source_lines, target_lines)
    matcher_blocks = list(matcher.get_matching_blocks())
    self.assertEqual(matcher_blocks, delta_blocks)
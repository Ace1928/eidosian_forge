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
def test_iter_lines_added_or_present_in_versions(self):

    class InstrumentedProgress(progress.ProgressTask):

        def __init__(self):
            progress.ProgressTask.__init__(self)
            self.updates = []

        def update(self, msg=None, current=None, total=None):
            self.updates.append((msg, current, total))
    vf = self.get_file()
    vf.add_lines(b'base', [], [b'base\n'])
    vf.add_lines(b'lancestor', [], [b'lancestor\n'])
    vf.add_lines(b'rancestor', [b'base'], [b'rancestor\n'])
    vf.add_lines(b'child', [b'rancestor'], [b'base\n', b'child\n'])
    vf.add_lines(b'otherchild', [b'lancestor', b'base'], [b'base\n', b'lancestor\n', b'otherchild\n'])

    def iter_with_versions(versions, expected):
        lines = {}
        progress = InstrumentedProgress()
        for line in vf.iter_lines_added_or_present_in_versions(versions, pb=progress):
            lines.setdefault(line, 0)
            lines[line] += 1
        if [] != progress.updates:
            self.assertEqual(expected, progress.updates)
        return lines
    lines = iter_with_versions([b'child', b'otherchild'], [('Walking content', 0, 2), ('Walking content', 1, 2), ('Walking content', 2, 2)])
    self.assertTrue(lines[b'child\n', b'child'] > 0)
    self.assertTrue(lines[b'otherchild\n', b'otherchild'] > 0)
    lines = iter_with_versions(None, [('Walking content', 0, 5), ('Walking content', 1, 5), ('Walking content', 2, 5), ('Walking content', 3, 5), ('Walking content', 4, 5), ('Walking content', 5, 5)])
    self.assertTrue(lines[b'base\n', b'base'] > 0)
    self.assertTrue(lines[b'lancestor\n', b'lancestor'] > 0)
    self.assertTrue(lines[b'rancestor\n', b'rancestor'] > 0)
    self.assertTrue(lines[b'child\n', b'child'] > 0)
    self.assertTrue(lines[b'otherchild\n', b'otherchild'] > 0)
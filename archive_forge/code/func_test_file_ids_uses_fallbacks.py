import sys
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy import tests, transform
from breezy.bzr import inventory, remote
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_file_ids_uses_fallbacks(self):
    builder = self.make_branch_builder('source', format=self.bzrdir_format)
    repo = builder.get_branch().repository
    if not repo._format.supports_external_lookups:
        raise tests.TestNotApplicable('format does not support stacking')
    builder.start_series()
    builder.build_snapshot(None, [('add', ('', b'root-id', 'directory', None)), ('add', ('file', b'file-id', 'file', b'contents\n'))], revision_id=b'A-id')
    builder.build_snapshot([b'A-id'], [('modify', ('file', b'new-content\n'))], revision_id=b'B-id')
    builder.build_snapshot([b'B-id'], [('modify', ('file', b'yet more content\n'))], revision_id=b'C-id')
    builder.finish_series()
    source_b = builder.get_branch()
    source_b.lock_read()
    self.addCleanup(source_b.unlock)
    base = self.make_branch('base')
    base.pull(source_b, stop_revision=b'B-id')
    stacked = self.make_branch('stacked')
    stacked.set_stacked_on_url('../base')
    stacked.pull(source_b, stop_revision=b'C-id')
    stacked.lock_read()
    self.addCleanup(stacked.unlock)
    repo = stacked.repository
    keys = {b'file-id': {b'A-id'}}
    if stacked.repository.supports_rich_root():
        keys[b'root-id'] = {b'A-id'}
    self.assertEqual(keys, repo.fileids_altered_by_revision_ids([b'A-id']))
import sys
from breezy import branch, controldir, errors, memorytree, tests
from breezy.bzr import branch as bzrbranch
from breezy.bzr import remote, versionedfile
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_get_missing_parent_inventories_check(self):
    builder = self.make_branch_builder('test')
    builder.build_snapshot([b'ghost-parent-id'], [('add', ('', b'root-id', 'directory', None)), ('add', ('file', b'file-id', 'file', b'content\n'))], allow_leftmost_as_ghost=True, revision_id=b'A-id')
    b = builder.get_branch()
    b.lock_read()
    self.addCleanup(b.unlock)
    repo = self.make_repository('test-repo')
    repo.lock_write()
    self.addCleanup(repo.unlock)
    repo.start_write_group()
    self.addCleanup(repo.abort_write_group)
    text_keys = [(b'file-id', b'A-id')]
    if repo.supports_rich_root():
        text_keys.append((b'root-id', b'A-id'))
    repo.texts.insert_record_stream(b.repository.texts.get_record_stream(text_keys, 'unordered', True))
    repo.add_revision(b'A-id', b.repository.get_revision(b'A-id'), b.repository.get_inventory(b'A-id'))
    get_missing = repo.get_missing_parent_inventories
    if repo._format.supports_external_lookups:
        self.assertEqual({('inventories', b'ghost-parent-id')}, get_missing(check_for_missing_texts=False))
        self.assertEqual(set(), get_missing(check_for_missing_texts=True))
        self.assertEqual(set(), get_missing())
    else:
        self.assertEqual(set(), get_missing(check_for_missing_texts=False))
        self.assertEqual(set(), get_missing(check_for_missing_texts=True))
        self.assertEqual(set(), get_missing())
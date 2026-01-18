import errno
from stat import S_ISDIR
import breezy.branch
from breezy import controldir, errors, repository
from breezy import revision as _mod_revision
from breezy import transport, workingtree
from breezy.bzr import bzrdir
from breezy.bzr.remote import RemoteBzrDirFormat
from breezy.bzr.tests.per_bzrdir import TestCaseWithBzrDir
from breezy.tests import TestNotApplicable, TestSkipped
from breezy.transport import FileExists
from breezy.transport.local import LocalTransport
def assertRepositoryHasSameItems(self, left_repo, right_repo):
    """require left_repo and right_repo to contain the same data."""
    with left_repo.lock_read(), right_repo.lock_read():
        all_revs = left_repo.all_revision_ids()
        self.assertEqual(left_repo.all_revision_ids(), right_repo.all_revision_ids())
        for rev_id in left_repo.all_revision_ids():
            self.assertEqual(left_repo.get_revision(rev_id), right_repo.get_revision(rev_id))

        def sort_key(rev_tree):
            return rev_tree.get_revision_id()
        rev_trees_a = sorted(left_repo.revision_trees(all_revs), key=sort_key)
        rev_trees_b = sorted(right_repo.revision_trees(all_revs), key=sort_key)
        for tree_a, tree_b in zip(rev_trees_a, rev_trees_b):
            self.assertEqual([], list(tree_a.iter_changes(tree_b)))
        text_index = left_repo._generate_text_key_index()
        self.assertEqual(text_index, right_repo._generate_text_key_index())
        desired_files = []
        for file_id, revision_id in text_index:
            desired_files.append((file_id, revision_id, (file_id, revision_id)))
        left_texts = [(identifier, b''.join(bytes_iterator)) for identifier, bytes_iterator in left_repo.iter_files_bytes(desired_files)]
        right_texts = [(identifier, b''.join(bytes_iterator)) for identifier, bytes_iterator in right_repo.iter_files_bytes(desired_files)]
        left_texts.sort()
        right_texts.sort()
        self.assertEqual(left_texts, right_texts)
        for rev_id in all_revs:
            try:
                left_text = left_repo.get_signature_text(rev_id)
            except errors.NoSuchRevision:
                continue
            right_text = right_repo.get_signature_text(rev_id)
            self.assertEqual(left_text, right_text)
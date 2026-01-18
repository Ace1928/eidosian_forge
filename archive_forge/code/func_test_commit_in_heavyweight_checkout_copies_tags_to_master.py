from breezy import branch as _mod_branch
from breezy import errors, lockable_files, lockdir, tag
from breezy.branch import Branch
from breezy.bzr import branch as bzrbranch
from breezy.bzr import bzrdir
from breezy.tests import TestCaseWithTransport, script
from breezy.workingtree import WorkingTree
def test_commit_in_heavyweight_checkout_copies_tags_to_master(self):
    master, child = self.make_master_and_checkout()
    fork = self.make_fork(master)
    fork.tags.set_tag('new-tag', fork.last_revision())
    fork.tags.set_tag('non-ancestry-tag', b'fork-0')
    fork.tags.set_tag('absent-tag', b'absent-rev')
    script.run_script(self, '\n            $ cd child\n            $ brz merge ../fork\n            $ brz commit -m "Merge fork."\n            2>Committing to: .../master/\n            2>Committed revision 2.\n            ', null_output_matches_anything=True)
    expected_tag_dict = {'new-tag': fork.last_revision(), 'non-ancestry-tag': b'fork-0', 'absent-tag': b'absent-rev'}
    self.assertEqual(expected_tag_dict, child.branch.tags.get_tag_dict())
    self.assertEqual(expected_tag_dict, master.tags.get_tag_dict())
    child.branch.repository.get_revision(b'fork-0')
    master.repository.get_revision(b'fork-0')
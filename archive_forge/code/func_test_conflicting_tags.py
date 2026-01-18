from breezy import branch as _mod_branch
from breezy import errors, lockable_files, lockdir, tag
from breezy.branch import Branch
from breezy.bzr import branch as bzrbranch
from breezy.bzr import bzrdir
from breezy.tests import TestCaseWithTransport, script
from breezy.workingtree import WorkingTree
def test_conflicting_tags(self):
    t1 = self.make_branch_and_tree('one')
    t2 = self.make_branch_and_tree('two')
    b1 = t1.branch
    b2 = t2.branch
    tagname = 'バzaar'
    b1.tags.set_tag(tagname, b'revid1')
    b2.tags.set_tag(tagname, b'revid2')
    out, err = self.run_bzr('push -d one two', encoding='utf-8')
    self.assertContainsRe(out, 'Conflicting tags:\n.*' + tagname)
    out, err = self.run_bzr('pull -d one two', encoding='utf-8', retcode=1)
    self.assertContainsRe(out, 'Conflicting tags:\n.*' + tagname)
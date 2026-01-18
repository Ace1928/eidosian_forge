import breezy.branch
from breezy import branch as _mod_branch
from breezy import check, controldir, errors, gpg, osutils
from breezy import repository as _mod_repository
from breezy import revision as _mod_revision
from breezy import transport, ui, urlutils, workingtree
from breezy.bzr import bzrdir as _mod_bzrdir
from breezy.bzr.remote import (RemoteBzrDir, RemoteBzrDirFormat,
from breezy.tests import (ChrootedTestCase, TestNotApplicable, TestSkipped,
from breezy.tests.per_controldir import TestCaseWithControlDir
from breezy.transport.local import LocalTransport
from breezy.ui import CannedInputUIFactory
def test_break_lock_branch(self):
    master = self.make_branch('branch')
    thisdir = self.make_controldir('this')
    try:
        thisdir.set_branch_reference(master)
    except errors.IncompatibleFormat:
        raise TestNotApplicable('format does not support branch references')
    unused_repo = thisdir.create_repository()
    master.lock_write()
    with unused_repo.lock_write():
        breezy.ui.ui_factory = CannedInputUIFactory([True, True, True])
        this_repo_locked = thisdir.find_repository().get_physical_lock_status()
        try:
            master.controldir.break_lock()
        except NotImplementedError:
            raise TestNotApplicable('format does not support breaking locks')
        if this_repo_locked:
            self.assertEqual([True], breezy.ui.ui_factory.responses)
        else:
            self.assertEqual([True, True], breezy.ui.ui_factory.responses)
        branch = master.controldir.open_branch()
        branch.lock_write()
        branch.unlock()
        if this_repo_locked:
            repo = thisdir.open_repository()
            self.assertRaises(errors.LockContention, repo.lock_write)
    self.assertRaises(errors.LockBroken, master.unlock)
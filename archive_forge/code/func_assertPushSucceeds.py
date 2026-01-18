import re
from breezy import (branch, controldir, directory_service, errors, osutils,
from breezy.bzr import bzrdir, knitrepo
from breezy.tests import http_server, scenarios, script, test_foreign
from breezy.transport import memory
def assertPushSucceeds(self, args, with_warning=False, revid_to_push=None):
    if with_warning:
        error_regexes = self._default_errors
    else:
        error_regexes = []
    out, err = self.run_bzr(self._default_command + args, working_dir=self._default_wd, error_regexes=error_regexes)
    if with_warning:
        self.assertContainsRe(err, self._default_additional_warning)
    else:
        self.assertNotContainsRe(err, self._default_additional_warning)
    branch_from = branch.Branch.open(self._default_wd)
    if revid_to_push is None:
        revid_to_push = branch_from.last_revision()
    branch_to = branch.Branch.open('to')
    repo_to = branch_to.repository
    self.assertTrue(repo_to.has_revision(revid_to_push))
    self.assertEqual(revid_to_push, branch_to.last_revision())
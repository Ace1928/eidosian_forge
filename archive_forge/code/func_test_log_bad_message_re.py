import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def test_log_bad_message_re(self):
    """Bad --message argument gives a sensible message

        See https://bugs.launchpad.net/bzr/+bug/251352
        """
    self.make_minimal_branch()
    out, err = self.run_bzr(['log', '-m', '*'], retcode=3)
    self.assertContainsRe(err, 'ERROR.*Invalid pattern.*nothing to repeat')
    self.assertNotContainsRe(err, 'Unprintable exception')
    self.assertEqual(out, '')
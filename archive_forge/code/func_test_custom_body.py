from unittest import TestLoader
from .... import config, tests
from ....bzr.bzrdir import BzrDir
from ....tests import TestCaseInTempDir
from ..emailer import EmailSender
def test_custom_body(self):
    sender, revid = self.get_sender(customized_mail_config)
    self.assertEqual('%s has committed revision 1 at %s.\n\n%s' % (sender.revision.committer, sender.url(), sample_log % revid.decode('utf-8')), sender.body())
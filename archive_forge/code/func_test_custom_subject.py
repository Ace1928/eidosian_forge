from unittest import TestLoader
from .... import config, tests
from ....bzr.bzrdir import BzrDir
from ....tests import TestCaseInTempDir
from ..emailer import EmailSender
def test_custom_subject(self):
    sender, revid = self.get_sender(customized_mail_config)
    self.assertEqual('[commit] %s' % sender.revision.get_summary(), sender.subject())
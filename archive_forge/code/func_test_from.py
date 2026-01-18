from unittest import TestLoader
from .... import config, tests
from ....bzr.bzrdir import BzrDir
from ....tests import TestCaseInTempDir
from ..emailer import EmailSender
def test_from(self):
    sender, revid = self.get_sender()
    self.assertEqual('Sample <foo@example.com>', sender.from_address())
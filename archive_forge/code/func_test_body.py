from unittest import TestLoader
from .... import config, tests
from ....bzr.bzrdir import BzrDir
from ....tests import TestCaseInTempDir
from ..emailer import EmailSender
def test_body(self):
    sender, revid = self.get_sender()
    self.assertEqual('At {}\n\n{}'.format(sender.url(), sample_log % revid.decode('utf-8')), sender.body())
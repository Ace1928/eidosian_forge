import os
import smtplib
from breezy import gpg, merge_directive, tests, workingtree
def test_mail_default(self):
    tree1, tree2 = self.prepare_merge_directive()
    md_text, errr, connect_calls, sendmail_calls = self.run_bzr_fakemail(['merge-directive', '--mail-to', 'pqm@example.com', '--plain', '../tree2', '.'])
    self.assertEqual('', md_text)
    self.assertEqual(1, len(connect_calls))
    call = connect_calls[0]
    self.assertEqual(('localhost', 0), call[1:3])
    self.assertEqual(1, len(sendmail_calls))
    call = sendmail_calls[0]
    self.assertEqual(('jrandom@example.com', ['pqm@example.com']), call[1:3])
    self.assertContainsRe(call[3], EMAIL1)
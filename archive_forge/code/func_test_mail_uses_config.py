import os
import smtplib
from breezy import gpg, merge_directive, tests, workingtree
def test_mail_uses_config(self):
    tree1, tree2 = self.prepare_merge_directive()
    br = tree1.branch
    br.get_config_stack().set('smtp_server', 'bogushost')
    md_text, errr, connect_calls, sendmail_calls = self.run_bzr_fakemail('merge-directive --mail-to pqm@example.com --plain ../tree2 .')
    call = connect_calls[0]
    self.assertEqual(('bogushost', 0), call[1:3])
import os
import smtplib
from breezy import gpg, merge_directive, tests, workingtree
def test_merge_directive(self):
    self.prepare_merge_directive()
    md_text = self.run_bzr('merge-directive ../tree2')[0]
    self.assertContainsRe(md_text, '\\+e')
    md_text = self.run_bzr('merge-directive -r -2 ../tree2')[0]
    self.assertNotContainsRe(md_text, '\\+e')
    md_text = self.run_bzr('merge-directive -r -1..-2 ../tree2')[0].encode('utf-8')
    md2 = merge_directive.MergeDirective.from_lines(md_text.splitlines(True))
    self.assertEqual(b'foo-id', md2.revision_id)
    self.assertEqual(b'bar-id', md2.base_revision_id)
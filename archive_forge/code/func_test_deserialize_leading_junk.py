import re
from .. import errors, gpg, mail_client, merge_directive, tests, trace
def test_deserialize_leading_junk(self):
    md = merge_directive.MergeDirective.from_lines(self.INPUT1)
    self.assertEqual(b'example:', md.revision_id)
    self.assertEqual(b'sha', md.testament_sha1)
    self.assertEqual('http://example.com', md.target_branch)
    self.assertEqual('http://example.org', md.source_branch)
    self.assertEqual(453, md.time)
    self.assertEqual(120, md.timezone)
    self.assertEqual(b'booga', md.patch)
    self.assertEqual('diff', md.patch_type)
    self.assertEqual('Hi mom!', md.message)
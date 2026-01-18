import re
from .. import errors, gpg, mail_client, merge_directive, tests, trace
def test_merge_source(self):
    time = 500000.0
    timezone = 5 * 3600
    self.assertRaises(errors.NoMergeSource, self.make_merge_directive, b'example:', b'sha', time, timezone, 'http://example.com')
    self.assertRaises(errors.NoMergeSource, self.make_merge_directive, b'example:', b'sha', time, timezone, 'http://example.com', patch_type='diff')
    self.make_merge_directive(b'example:', b'sha', time, timezone, 'http://example.com', source_branch='http://example.org')
    md = self.make_merge_directive(b'null:', b'sha', time, timezone, 'http://example.com', patch=b'blah', patch_type='bundle')
    self.assertIs(None, md.source_branch)
    md2 = self.make_merge_directive(b'null:', b'sha', time, timezone, 'http://example.com', patch=b'blah', patch_type='bundle', source_branch='bar')
    self.assertEqual('bar', md2.source_branch)
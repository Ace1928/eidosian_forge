import re
from .. import errors, gpg, mail_client, merge_directive, tests, trace
def test_deserialize_junk(self):
    time = 501
    self.assertRaises(errors.NotAMergeDirective, merge_directive.MergeDirective.from_lines, [b'lala'])
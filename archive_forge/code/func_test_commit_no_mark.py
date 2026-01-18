from unittest import TestCase
from fastimport.helpers import (
from fastimport import (
def test_commit_no_mark(self):
    committer = (b'Joe Wong', b'joe@example.com', 1234567890, -6 * 3600)
    c = commands.CommitCommand(b'refs/heads/master', None, None, committer, b'release v1.0', b':aaa', None, None)
    self.assertEqual(b'commit refs/heads/master\ncommitter Joe Wong <joe@example.com> 1234567890 -0600\ndata 12\nrelease v1.0\nfrom :aaa', bytes(c))
from unittest import TestCase
from fastimport.helpers import (
from fastimport import (
def test_commit_with_filecommands(self):
    file_cmds = iter([commands.FileDeleteCommand(b'readme.txt'), commands.FileModifyCommand(b'NEWS', 33188, None, b'blah blah blah')])
    committer = (b'Joe Wong', b'joe@example.com', 1234567890, -6 * 3600)
    c = commands.CommitCommand(b'refs/heads/master', b'bbb', None, committer, b'release v1.0', b':aaa', None, file_cmds)
    self.assertEqual(b'commit refs/heads/master\nmark :bbb\ncommitter Joe Wong <joe@example.com> 1234567890 -0600\ndata 12\nrelease v1.0\nfrom :aaa\nD readme.txt\nM 644 inline NEWS\ndata 14\nblah blah blah', bytes(c))
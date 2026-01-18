from dulwich.objects import Blob, Commit, Tag, parse_timezone
from dulwich.tests.utils import make_object
from ...revision import Revision
from .. import tests
from ..mapping import (BzrGitMappingv1, UnknownCommitEncoding,
def test_commit_double_negative_timezone(self):
    c = Commit()
    c.tree = b'cc9462f7f8263ef5adfbeff2fb936bb36b504cba'
    c.message = b'Some message'
    c.committer = b'Committer <Committer>'
    c.commit_time = 4
    c.commit_timezone, c._commit_timezone_neg_utc = parse_timezone(b'--700')
    c.author_time = 5
    c.author_timezone = 60 * 2
    c.author = b'Author <author>'
    self.assertRoundtripCommit(c)
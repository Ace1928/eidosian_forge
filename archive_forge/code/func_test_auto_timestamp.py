from io import BytesIO
from dulwich.repo import Repo as GitRepo
from .. import tests
def test_auto_timestamp(self):
    stream = BytesIO()
    builder = tests.GitBranchBuilder(stream)
    builder.commit(b'Joe Foo <joe@foo.com>', 'message')
    self.assertContainsRe(stream.getvalue(), b'committer Joe Foo <joe@foo\\.com> \\d+ \\+0000')
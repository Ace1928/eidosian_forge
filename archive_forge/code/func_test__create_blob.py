from io import BytesIO
from dulwich.repo import Repo as GitRepo
from .. import tests
def test__create_blob(self):
    stream = BytesIO()
    builder = tests.GitBranchBuilder(stream)
    self.assertEqual(1, builder._create_blob(b'foo\nbar\n'))
    self.assertEqualDiff(b'blob\nmark :1\ndata 8\nfoo\nbar\n\n', stream.getvalue())
from io import BytesIO
from dulwich.repo import Repo as GitRepo
from .. import tests
def test_delete_entry_newline(self):
    stream = BytesIO()
    builder = tests.GitBranchBuilder(stream)
    builder.delete_entry('path/to/foo\nbar')
    self.assertEqual([b'D "path/to/foo\\nbar"\n'], builder.commit_info)
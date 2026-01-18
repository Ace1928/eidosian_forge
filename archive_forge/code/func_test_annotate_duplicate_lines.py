import codecs
from io import BytesIO, StringIO
from .. import annotate, tests
from .ui_testing import StringIOWithEncoding
def test_annotate_duplicate_lines(self):
    builder = self.create_duplicate_lines_tree()
    repo = builder.get_branch().repository
    repo.lock_read()
    self.addCleanup(repo.unlock)
    self.assertRepoAnnotate(duplicate_base, repo, 'file', b'rev-base')
    self.assertRepoAnnotate(duplicate_A, repo, 'file', b'rev-A')
    self.assertRepoAnnotate(duplicate_B, repo, 'file', b'rev-B')
    self.assertRepoAnnotate(duplicate_C, repo, 'file', b'rev-C')
    self.assertRepoAnnotate(duplicate_D, repo, 'file', b'rev-D')
    self.assertRepoAnnotate(duplicate_E, repo, 'file', b'rev-E')
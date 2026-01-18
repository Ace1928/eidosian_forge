from .. import annotate, errors, revision, tests
from ..bzr import knit
def make_common_merge_text(self):
    """Both sides of the merge will have introduced a line."""
    self.make_simple_text()
    self.vf.add_lines(self.fc_key, [self.fa_key], [b'simple\n', b'new content\n'])
    self.vf.add_lines(self.fd_key, [self.fb_key, self.fc_key], [b'simple\n', b'new content\n'])
from .. import annotate, errors, revision, tests
from ..bzr import knit
def make_merge_text(self):
    self.make_simple_text()
    self.vf.add_lines(self.fc_key, [self.fa_key], [b'simple\n', b'from c\n', b'content\n'])
    self.vf.add_lines(self.fd_key, [self.fb_key, self.fc_key], [b'simple\n', b'from c\n', b'new content\n', b'introduced in merge\n'])
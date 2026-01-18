from .. import annotate, errors, revision, tests
from ..bzr import knit
def test_annotate_common_merge_text(self):
    self.make_common_merge_text()
    self.assertAnnotateEqual([(self.fa_key,), (self.fb_key, self.fc_key)], self.fd_key)
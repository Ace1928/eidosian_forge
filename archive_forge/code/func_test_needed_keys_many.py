from .. import annotate, errors, revision, tests
from ..bzr import knit
def test_needed_keys_many(self):
    self.make_many_way_common_merge_text()
    keys, ann_keys = self.ann._get_needed_keys(self.ff_key)
    self.assertEqual([self.fa_key, self.fb_key, self.fc_key, self.fd_key, self.fe_key, self.ff_key], sorted(keys))
    self.assertEqual({self.fa_key: 3, self.fb_key: 1, self.fc_key: 1, self.fd_key: 1, self.fe_key: 1, self.ff_key: 1}, self.ann._num_needed_children)
    self.assertEqual(set(), ann_keys)
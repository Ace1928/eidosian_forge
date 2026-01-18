from .. import annotate, errors, revision, tests
from ..bzr import knit
def test_needed_keys_with_parent_texts(self):
    self.make_many_way_common_merge_text()
    self.ann._parent_map[self.fd_key] = (self.fb_key, self.fc_key)
    self.ann._text_cache[self.fd_key] = [b'simple\n', b'new content\n']
    self.ann._annotations_cache[self.fd_key] = [(self.fa_key,), (self.fb_key, self.fc_key)]
    self.ann._parent_map[self.fe_key] = (self.fa_key,)
    self.ann._text_cache[self.fe_key] = [b'simple\n', b'new content\n']
    self.ann._annotations_cache[self.fe_key] = [(self.fa_key,), (self.fe_key,)]
    keys, ann_keys = self.ann._get_needed_keys(self.ff_key)
    self.assertEqual([self.ff_key], sorted(keys))
    self.assertEqual({self.fd_key: 1, self.fe_key: 1, self.ff_key: 1}, self.ann._num_needed_children)
    self.assertEqual([], sorted(ann_keys))
from .. import annotate, errors, revision, tests
from ..bzr import knit
def test_annotate_many_way_common_merge_text_more(self):
    self.make_many_way_common_merge_text()
    self.assertEqual([(self.fa_key, b'simple\n'), (self.fb_key, b'new content\n')], self.ann.annotate_flat(self.ff_key))
from .. import annotate, errors, revision, tests
from ..bzr import knit
def make_no_graph_texts(self):
    factory = knit.make_pack_factory(False, False, 2)
    self.vf = factory(self.get_transport())
    self.ann = self.module.Annotator(self.vf)
    self.vf.add_lines(self.fa_key, (), [b'simple\n', b'content\n'])
    self.vf.add_lines(self.fb_key, (), [b'simple\n', b'new content\n'])
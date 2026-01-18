from ... import errors, multiparent, tests
from .. import groupcompress, versionedfile
def make_three_vf(self):
    vf = self.make_vf()
    vf.add_lines((b'one',), (), [b'first\n'])
    vf.add_lines((b'two',), [(b'one',)], [b'first\n', b'second\n'])
    vf.add_lines((b'three',), [(b'one',), (b'two',)], [b'first\n', b'second\n', b'third\n'])
    return vf
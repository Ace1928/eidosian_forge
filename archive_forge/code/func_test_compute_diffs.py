from ... import errors, multiparent, tests
from .. import groupcompress, versionedfile
def test_compute_diffs(self):
    vf = self.make_three_vf()
    gen = versionedfile._MPDiffGenerator(vf, [(b'two',), (b'three',), (b'one',)])
    diffs = gen.compute_diffs()
    expected_diffs = [multiparent.MultiParent([multiparent.ParentText(0, 0, 0, 1), multiparent.NewText([b'second\n'])]), multiparent.MultiParent([multiparent.ParentText(1, 0, 0, 2), multiparent.NewText([b'third\n'])]), multiparent.MultiParent([multiparent.NewText([b'first\n'])])]
    self.assertEqual(expected_diffs, diffs)
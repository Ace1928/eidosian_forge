from ... import errors, multiparent, tests
from .. import groupcompress, versionedfile
def test_finds_parents(self):
    vf = self.make_three_vf()
    gen = versionedfile._MPDiffGenerator(vf, [(b'three',)])
    needed_keys, refcount = gen._find_needed_keys()
    self.assertEqual(sorted([(b'one',), (b'two',), (b'three',)]), sorted(needed_keys))
    self.assertEqual({(b'one',): 1, (b'two',): 1}, refcount)
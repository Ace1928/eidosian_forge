import sys
from .. import branch as _mod_branch
from .. import controldir, errors, info
from .. import repository as _mod_repository
from .. import tests, workingtree
from ..bzr import branch as _mod_bzrbranch
def test_location_list(self):
    if sys.platform == 'win32':
        raise tests.TestSkipped('Windows-unfriendly test')
    locs = info.LocationList('/home/foo')
    locs.add_url('a', 'file:///home/foo/')
    locs.add_url('b', 'file:///home/foo/bar/')
    locs.add_url('c', 'file:///home/bar/bar')
    locs.add_url('d', 'http://example.com/example/')
    locs.add_url('e', None)
    self.assertEqual(locs.locs, [('a', '.'), ('b', 'bar'), ('c', '/home/bar/bar'), ('d', 'http://example.com/example/')])
    self.assertEqualDiff('  a: .\n  b: bar\n  c: /home/bar/bar\n  d: http://example.com/example/\n', ''.join(locs.get_lines()))
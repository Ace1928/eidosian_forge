import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_add_sorting(self):
    """Add entries in lexicographical order, we get path sorted order.

        This tests it to a depth of 4, to make sure we don't just get it right
        at a single depth. 'a/a' should come before 'a-a', even though it
        doesn't lexicographically.
        """
    dirs = ['a', 'a/a', 'a/a/a', 'a/a/a/a', 'a-a', 'a/a-a', 'a/a/a-a', 'a/a/a/a-a']
    null_sha = b''
    state = dirstate.DirState.initialize('dirstate')
    self.addCleanup(state.unlock)
    fake_stat = os.stat('dirstate')
    for d in dirs:
        d_id = d.encode('utf-8').replace(b'/', b'_') + b'-id'
        file_path = d + '/f'
        file_id = file_path.encode('utf-8').replace(b'/', b'_') + b'-id'
        state.add(d, d_id, 'directory', fake_stat, null_sha)
        state.add(file_path, file_id, 'file', fake_stat, null_sha)
    expected = [b'', b'', b'a', b'a/a', b'a/a/a', b'a/a/a/a', b'a/a/a/a-a', b'a/a/a-a', b'a/a-a', b'a-a']

    def split(p):
        return p.split(b'/')
    self.assertEqual(sorted(expected, key=split), expected)
    dirblock_names = [d[0] for d in state._dirblocks]
    self.assertEqual(expected, dirblock_names)
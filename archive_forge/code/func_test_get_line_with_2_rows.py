import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_get_line_with_2_rows(self):
    state = self.create_dirstate_with_root_and_subdir()
    try:
        self.assertEqual([b'#bazaar dirstate flat format 3\n', b'crc32: 41262208\n', b'num_entries: 2\n', b'0\x00\n\x000\x00\n\x00\x00\x00a-root-value\x00d\x00\x000\x00n\x00AAAAREUHaIpFB2iKAAADAQAtkqUAAIGk\x00\n\x00\x00subdir\x00subdir-id\x00d\x00\x000\x00n\x00AAAAREUHaIpFB2iKAAADAQAtkqUAAIGk\x00\n\x00'], state.get_lines())
    finally:
        state.unlock()
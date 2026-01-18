import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
@staticmethod
def from_stat(st):
    return _FakeStat(st.st_size, st.st_mtime, st.st_ctime, st.st_dev, st.st_ino, st.st_mode)
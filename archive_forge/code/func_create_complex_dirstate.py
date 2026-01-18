import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def create_complex_dirstate(self):
    """This dirstate contains multiple files and directories.

         /        a-root-value
         a/       a-dir
         b/       b-dir
         c        c-file
         d        d-file
         a/e/     e-dir
         a/f      f-file
         b/g      g-file
         b/hÃ¥  h-Ã¥-file  #This is u'å' encoded into utf-8

        Notice that a/e is an empty directory.

        :return: The dirstate, still write-locked.
        """
    packed_stat = b'AAAAREUHaIpFB2iKAAADAQAtkqUAAIGk'
    null_sha = b'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
    root_entry = ((b'', b'', b'a-root-value'), [(b'd', b'', 0, False, packed_stat)])
    a_entry = ((b'', b'a', b'a-dir'), [(b'd', b'', 0, False, packed_stat)])
    b_entry = ((b'', b'b', b'b-dir'), [(b'd', b'', 0, False, packed_stat)])
    c_entry = ((b'', b'c', b'c-file'), [(b'f', null_sha, 10, False, packed_stat)])
    d_entry = ((b'', b'd', b'd-file'), [(b'f', null_sha, 20, False, packed_stat)])
    e_entry = ((b'a', b'e', b'e-dir'), [(b'd', b'', 0, False, packed_stat)])
    f_entry = ((b'a', b'f', b'f-file'), [(b'f', null_sha, 30, False, packed_stat)])
    g_entry = ((b'b', b'g', b'g-file'), [(b'f', null_sha, 30, False, packed_stat)])
    h_entry = ((b'b', b'h\xc3\xa5', b'h-\xc3\xa5-file'), [(b'f', null_sha, 40, False, packed_stat)])
    dirblocks = []
    dirblocks.append((b'', [root_entry]))
    dirblocks.append((b'', [a_entry, b_entry, c_entry, d_entry]))
    dirblocks.append((b'a', [e_entry, f_entry]))
    dirblocks.append((b'b', [g_entry, h_entry]))
    state = dirstate.DirState.initialize('dirstate')
    state._validate()
    try:
        state._set_data([], dirblocks)
    except:
        state.unlock()
        raise
    return state
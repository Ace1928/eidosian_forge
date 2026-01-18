import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def create_dirstate_with_two_trees(self):
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

        There is one parent tree, which has the same shape with the following variations:
        b/g in the parent is gone.
        b/h in the parent has a different id
        b/i is new in the parent
        c is renamed to b/j in the parent

        :return: The dirstate, still write-locked.
        """
    packed_stat = b'AAAAREUHaIpFB2iKAAADAQAtkqUAAIGk'
    null_sha = b'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
    NULL_PARENT_DETAILS = dirstate.DirState.NULL_PARENT_DETAILS
    root_entry = ((b'', b'', b'a-root-value'), [(b'd', b'', 0, False, packed_stat), (b'd', b'', 0, False, b'parent-revid')])
    a_entry = ((b'', b'a', b'a-dir'), [(b'd', b'', 0, False, packed_stat), (b'd', b'', 0, False, b'parent-revid')])
    b_entry = ((b'', b'b', b'b-dir'), [(b'd', b'', 0, False, packed_stat), (b'd', b'', 0, False, b'parent-revid')])
    c_entry = ((b'', b'c', b'c-file'), [(b'f', null_sha, 10, False, packed_stat), (b'r', b'b/j', 0, False, b'')])
    d_entry = ((b'', b'd', b'd-file'), [(b'f', null_sha, 20, False, packed_stat), (b'f', b'd', 20, False, b'parent-revid')])
    e_entry = ((b'a', b'e', b'e-dir'), [(b'd', b'', 0, False, packed_stat), (b'd', b'', 0, False, b'parent-revid')])
    f_entry = ((b'a', b'f', b'f-file'), [(b'f', null_sha, 30, False, packed_stat), (b'f', b'f', 20, False, b'parent-revid')])
    g_entry = ((b'b', b'g', b'g-file'), [(b'f', null_sha, 30, False, packed_stat), NULL_PARENT_DETAILS])
    h_entry1 = ((b'b', b'h\xc3\xa5', b'h-\xc3\xa5-file1'), [(b'f', null_sha, 40, False, packed_stat), NULL_PARENT_DETAILS])
    h_entry2 = ((b'b', b'h\xc3\xa5', b'h-\xc3\xa5-file2'), [NULL_PARENT_DETAILS, (b'f', b'h', 20, False, b'parent-revid')])
    i_entry = ((b'b', b'i', b'i-file'), [NULL_PARENT_DETAILS, (b'f', b'h', 20, False, b'parent-revid')])
    j_entry = ((b'b', b'j', b'c-file'), [(b'r', b'c', 0, False, b''), (b'f', b'j', 20, False, b'parent-revid')])
    dirblocks = []
    dirblocks.append((b'', [root_entry]))
    dirblocks.append((b'', [a_entry, b_entry, c_entry, d_entry]))
    dirblocks.append((b'a', [e_entry, f_entry]))
    dirblocks.append((b'b', [g_entry, h_entry1, h_entry2, i_entry, j_entry]))
    state = dirstate.DirState.initialize('dirstate')
    state._validate()
    try:
        state._set_data([b'parent'], dirblocks)
    except:
        state.unlock()
        raise
    return (state, dirblocks)
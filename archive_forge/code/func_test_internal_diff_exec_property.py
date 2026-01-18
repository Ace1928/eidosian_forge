import contextlib
import os
import re
import subprocess
import sys
import tempfile
from io import BytesIO
from .. import diff, errors, osutils
from .. import revision as _mod_revision
from .. import revisionspec, revisiontree, tests
from ..tests import EncodingAdapter, features
from ..tests.scenarios import load_tests_apply_scenarios
def test_internal_diff_exec_property(self):
    tree = self.make_branch_and_tree('tree')
    tt = tree.transform()
    tt.new_file('a', tt.root, [b'contents\n'], b'a-id', True)
    tt.new_file('b', tt.root, [b'contents\n'], b'b-id', False)
    tt.new_file('c', tt.root, [b'contents\n'], b'c-id', True)
    tt.new_file('d', tt.root, [b'contents\n'], b'd-id', False)
    tt.new_file('e', tt.root, [b'contents\n'], b'control-e-id', True)
    tt.new_file('f', tt.root, [b'contents\n'], b'control-f-id', False)
    tt.apply()
    tree.commit('one', rev_id=b'rev-1')
    tt = tree.transform()
    tt.set_executability(False, tt.trans_id_file_id(b'a-id'))
    tt.set_executability(True, tt.trans_id_file_id(b'b-id'))
    tt.set_executability(False, tt.trans_id_file_id(b'c-id'))
    tt.set_executability(True, tt.trans_id_file_id(b'd-id'))
    tt.apply()
    tree.rename_one('c', 'new-c')
    tree.rename_one('d', 'new-d')
    d = get_diff_as_string(tree.basis_tree(), tree)
    self.assertContainsRe(d, b"file 'a'.*\\(properties changed:.*\\+x to -x.*\\)")
    self.assertContainsRe(d, b"file 'b'.*\\(properties changed:.*-x to \\+x.*\\)")
    self.assertContainsRe(d, b"file 'c'.*\\(properties changed:.*\\+x to -x.*\\)")
    self.assertContainsRe(d, b"file 'd'.*\\(properties changed:.*-x to \\+x.*\\)")
    self.assertNotContainsRe(d, b"file 'e'")
    self.assertNotContainsRe(d, b"file 'f'")
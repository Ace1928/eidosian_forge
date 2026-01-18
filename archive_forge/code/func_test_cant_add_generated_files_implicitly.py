import os
import sys
from io import StringIO
from ... import add as _mod_add
from ... import errors, ignores, osutils, tests, trace, transport, workingtree
from .. import features, per_workingtree, test_smart_add
def test_cant_add_generated_files_implicitly(self):
    t = self.make_tree_with_text_conflict()
    added, ignored = t.smart_add([t.basedir])
    self.assertEqual(([], {}), (added, ignored))
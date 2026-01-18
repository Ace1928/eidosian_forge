import os
from .. import errors
from .. import merge as _mod_merge
from ..bzr.conflicts import TextConflict
from . import TestCaseWithTransport, multiply_tests
from .test_merge_core import MergeBuilder
def test_change_vs_change(self):
    """Hook is used for (changed, changed)"""
    self.install_hook_success()
    builder = self.make_merge_builder()
    name1 = builder.add_file(builder.root(), 'name1', b'text1', True, file_id=b'1')
    builder.change_contents(name1, other=b'text4', this=b'text3')
    conflicts = builder.merge(self.merge_type)
    self.assertEqual(conflicts, [])
    with builder.this.get_file('name1') as f:
        self.assertEqual(f.read(), b'text-merged-by-hook')
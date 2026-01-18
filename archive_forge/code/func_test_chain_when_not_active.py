import os
from .. import errors
from .. import merge as _mod_merge
from ..bzr.conflicts import TextConflict
from . import TestCaseWithTransport, multiply_tests
from .test_merge_core import MergeBuilder
def test_chain_when_not_active(self):
    """When a hook function returns None, merging still works."""
    self.install_hook_inactive()
    self.install_hook_success()
    builder = self.make_merge_builder()
    self.create_file_needing_contents_merge(builder, 'name1')
    conflicts = builder.merge(self.merge_type)
    self.assertEqual(conflicts, [])
    with builder.this.get_file('name1') as f:
        self.assertEqual(f.read(), b'text-merged-by-hook')
    self.assertEqual([('inactive',), ('success',)], self.hook_log)
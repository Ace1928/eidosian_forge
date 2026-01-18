import os
from .. import errors
from .. import merge as _mod_merge
from ..bzr.conflicts import TextConflict
from . import TestCaseWithTransport, multiply_tests
from .test_merge_core import MergeBuilder
def test_chain_stops_after_success(self):
    """When a hook function returns success, no later functions are tried.
        """
    self.install_hook_success()
    self.install_hook_noop()
    builder = self.make_merge_builder()
    self.create_file_needing_contents_merge(builder, 'name1')
    conflicts = builder.merge(self.merge_type)
    self.assertEqual([('success',)], self.hook_log)
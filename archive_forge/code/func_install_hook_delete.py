import os
from .. import errors
from .. import merge as _mod_merge
from ..bzr.conflicts import TextConflict
from . import TestCaseWithTransport, multiply_tests
from .test_merge_core import MergeBuilder
def install_hook_delete(self):
    test = self

    class HookDelete(_mod_merge.AbstractPerFileMerger):

        def merge_contents(self, merge_params):
            test.hook_log.append(('delete',))
            if merge_params.this_path == 'name1':
                return ('delete', None)
            return ('not_applicable', None)

    def hook_delete_factory(merger):
        return HookDelete(merger)
    _mod_merge.Merger.hooks.install_named_hook('merge_file_content', hook_delete_factory, 'test hook (delete)')
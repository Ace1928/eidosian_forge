import os
from .. import errors
from .. import merge as _mod_merge
from ..bzr.conflicts import TextConflict
from . import TestCaseWithTransport, multiply_tests
from .test_merge_core import MergeBuilder
def merge_contents(self, merge_params):
    test.hook_log.append(('log_lines', merge_params.this_lines, merge_params.other_lines, merge_params.base_lines))
    return ('not_applicable', None)
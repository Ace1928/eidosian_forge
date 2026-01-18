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
def call_gtabtd(self, path_list, revision_specs, old_url, new_url):
    """Call get_trees_and_branches_to_diff_locked."""
    exit_stack = contextlib.ExitStack()
    self.addCleanup(exit_stack.close)
    return diff.get_trees_and_branches_to_diff_locked(path_list, revision_specs, old_url, new_url, exit_stack)
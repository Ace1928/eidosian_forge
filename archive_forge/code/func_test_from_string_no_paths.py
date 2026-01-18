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
def test_from_string_no_paths(self):
    diff_obj = diff.DiffFromTool.from_string(['diff', '-u5'], None, None, None)
    self.addCleanup(diff_obj.finish)
    self.assertEqual(['diff', '-u5'], diff_obj.command_template)
    self.assertEqual(['diff', '-u5', 'old-path', 'new-path'], diff_obj._get_command('old-path', 'new-path'))
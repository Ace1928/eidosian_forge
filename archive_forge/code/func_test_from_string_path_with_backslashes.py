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
def test_from_string_path_with_backslashes(self):
    self.requireFeature(features.backslashdir_feature)
    tool = ['C:\\Tools\\Diff.exe', '{old_path}', '{new_path}']
    diff_obj = diff.DiffFromTool.from_string(tool, None, None, None)
    self.addCleanup(diff_obj.finish)
    self.assertEqual(['C:\\Tools\\Diff.exe', '{old_path}', '{new_path}'], diff_obj.command_template)
    self.assertEqual(['C:\\Tools\\Diff.exe', 'old-path', 'new-path'], diff_obj._get_command('old-path', 'new-path'))
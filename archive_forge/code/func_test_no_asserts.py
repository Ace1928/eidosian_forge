import ast
import os
import re
import sys
import breezy.branch
from breezy import osutils
from breezy.tests import TestCase, TestSkipped, features
def test_no_asserts(self):
    """bzr shouldn't use the 'assert' statement."""
    badfiles = []
    assert_re = re.compile('\\bassert\\b')
    for fname, text in self.get_source_file_contents():
        if not self.is_our_code(fname):
            continue
        if not assert_re.search(text):
            continue
        tree = ast.parse(text)
        for entry in ast.walk(tree):
            if isinstance(entry, ast.Assert):
                badfiles.append(fname)
                break
    if badfiles:
        self.fail('these files contain an assert statement and should not:\n%s' % '\n'.join(badfiles))
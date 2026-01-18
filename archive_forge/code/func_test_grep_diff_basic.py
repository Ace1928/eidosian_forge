import os
import re
import unicodedata as ud
from .. import osutils, tests
from .._termcolor import FG, color_string
from ..tests.features import UnicodeFilenameFeature
def test_grep_diff_basic(self):
    """grep -p basic test."""
    tree = self.make_example_branch()
    self.build_tree_contents([('hello', b'hello world!\n')])
    tree.commit('updated hello')
    out, err = self.run_bzr(['grep', '-p', 'hello'])
    self.assertEqual(err, '')
    self.assertEqualDiff(subst_dates(out), "=== revno:3 ===\n  === modified file 'hello'\n    --- hello\tYYYY-MM-DD HH:MM:SS +ZZZZ\n    +++ hello\tYYYY-MM-DD HH:MM:SS +ZZZZ\n    +hello world!\n=== revno:1 ===\n  === added file 'hello'\n    --- hello\tYYYY-MM-DD HH:MM:SS +ZZZZ\n    +++ hello\tYYYY-MM-DD HH:MM:SS +ZZZZ\n")
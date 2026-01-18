import os
import re
import unicodedata as ud
from .. import osutils, tests
from .._termcolor import FG, color_string
from ..tests.features import UnicodeFilenameFeature
def test_grep_diff_revision_range(self):
    """grep -p revision range."""
    tree = self.make_example_branch()
    self.build_tree_contents([('hello', b'hello world!1\n')])
    tree.commit('rev3')
    self.build_tree_contents([('blah', b'hello world!2\n')])
    tree.add('blah')
    tree.commit('rev4')
    with open('hello', 'a') as f:
        f.write('hello world!3\n')
    tree.commit('rev5')
    out, err = self.run_bzr(['grep', '-p', '-r', '2..5', 'hello'])
    self.assertEqual(err, '')
    self.assertEqualDiff(subst_dates(out), "=== revno:5 ===\n  === modified file 'hello'\n    --- hello\tYYYY-MM-DD HH:MM:SS +ZZZZ\n    +++ hello\tYYYY-MM-DD HH:MM:SS +ZZZZ\n    +hello world!3\n=== revno:4 ===\n  === added file 'blah'\n    +hello world!2\n=== revno:3 ===\n  === modified file 'hello'\n    --- hello\tYYYY-MM-DD HH:MM:SS +ZZZZ\n    +++ hello\tYYYY-MM-DD HH:MM:SS +ZZZZ\n    +hello world!1\n")
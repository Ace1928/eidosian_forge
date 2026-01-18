import os
import re
import unicodedata as ud
from .. import osutils, tests
from .._termcolor import FG, color_string
from ..tests.features import UnicodeFilenameFeature
def test_grep_diff_color(self):
    """grep -p color test."""
    tree = self.make_example_branch()
    self.build_tree_contents([('hello', b'hello world!\n')])
    tree.commit('updated hello')
    out, err = self.run_bzr(['grep', '--diff', '-r', '3', '--color', 'always', 'hello'])
    self.assertEqual(err, '')
    revno = color_string('=== revno:3 ===', fg=FG.BOLD_BLUE) + '\n'
    filename = color_string("  === modified file 'hello'", fg=FG.BOLD_MAGENTA) + '\n'
    redhello = color_string('hello', fg=FG.BOLD_RED)
    diffstr = '    --- hello\tYYYY-MM-DD HH:MM:SS +ZZZZ\n    +++ hello\tYYYY-MM-DD HH:MM:SS +ZZZZ\n    +hello world!\n'
    diffstr = diffstr.replace('hello', redhello)
    self.assertEqualDiff(subst_dates(out), revno + filename + diffstr)
import os
import re
import unicodedata as ud
from .. import osutils, tests
from .._termcolor import FG, color_string
from ..tests.features import UnicodeFilenameFeature
def test_grep_norevs(self):
    """grep -p with zero revisions."""
    out, err = self.run_bzr(['init'])
    out, err = self.run_bzr(['grep', '--diff', 'foo'], 3)
    self.assertEqual(out, '')
    self.assertContainsRe(err, 'ERROR:.*revision.* does not exist in branch')
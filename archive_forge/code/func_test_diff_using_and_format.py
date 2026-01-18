import os
import re
from breezy import tests, workingtree
from breezy.diff import DiffTree
from breezy.diff import format_registry as diff_format_registry
from breezy.tests import features
def test_diff_using_and_format(self):
    out, err = self.run_bzr('diff --format=default --using=mydi', retcode=3, error_regexes=('are mutually exclusive',))
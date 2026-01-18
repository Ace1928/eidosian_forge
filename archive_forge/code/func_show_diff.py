import os
import re
from breezy import tests, workingtree
from breezy.diff import DiffTree
from breezy.diff import format_registry as diff_format_registry
from breezy.tests import features
def show_diff(self, specific_files, extra_trees=None):
    self.to_file.write('BOO!\n')
    return super().show_diff(specific_files, extra_trees)
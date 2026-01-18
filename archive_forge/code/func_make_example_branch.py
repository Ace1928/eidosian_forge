import os
import re
from breezy import tests, workingtree
from breezy.diff import DiffTree
from breezy.diff import format_registry as diff_format_registry
from breezy.tests import features
def make_example_branch(self):
    tree = super().make_example_branch()
    tree = tree.branch.create_checkout('checkout')
    os.chdir('checkout')
    return tree
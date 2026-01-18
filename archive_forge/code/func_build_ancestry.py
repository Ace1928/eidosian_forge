from __future__ import print_function
from __future__ import unicode_literals
import logging
from cmakelang import common
from cmakelang import lex
from cmakelang.parse.util import (
def build_ancestry(self):
    """Recursively assign the .parent member within the subtree."""
    for child in self.children:
        if isinstance(child, TreeNode):
            child.parent = self
            child.build_ancestry()
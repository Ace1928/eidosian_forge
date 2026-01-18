import os
import sys
from io import BytesIO
from textwrap import dedent
from .. import errors, revision, shelf, shelf_ui, tests
from . import features, script
def create_shelvable_tree(self):
    tree = self.make_branch_and_tree('tree')
    self.build_tree_contents([('tree/foo', LINES_AJ)])
    tree.add('foo', ids=b'foo-id')
    tree.commit('added foo')
    self.build_tree_contents([('tree/foo', LINES_ZY)])
    return tree
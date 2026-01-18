import os
import sys
from io import BytesIO
from textwrap import dedent
from .. import errors, revision, shelf, shelf_ui, tests
from . import features, script
def create_tree_with_shelf(self):
    tree = self.make_branch_and_tree('tree')
    tree.lock_write()
    try:
        self.build_tree_contents([('tree/foo', LINES_AJ)])
        tree.add('foo', ids=b'foo-id')
        tree.commit('added foo')
        self.build_tree_contents([('tree/foo', LINES_ZY)])
        shelver = shelf_ui.Shelver(tree, tree.basis_tree(), auto_apply=True, auto=True)
        try:
            shelver.run()
        finally:
            shelver.finalize()
    finally:
        tree.unlock()
    return tree
import os
from io import BytesIO
from .. import bedding, ignores
from . import TestCase, TestCaseInTempDir, TestCaseWithTransport
def test_adds_ending_newline(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree_contents([('.bzrignore', b'myentry1')])
    tree.add(['.bzrignore'])
    ignores.tree_ignores_add_patterns(tree, ['myentry2'])
    self.assertPatternsEquals(['myentry1', 'myentry2'])
    with open('.bzrignore') as f:
        text = f.read()
    self.assertTrue(text.endswith(('\r\n', '\n', '\r')))
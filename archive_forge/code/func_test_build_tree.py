import os
from breezy import tests
from breezy.tests import features
def test_build_tree(self):
    """Test tree-building test helper"""
    self.build_tree_contents([('foo', b'new contents'), ('.bzr/',), ('.bzr/README', b'hello')])
    self.assertPathExists('foo')
    self.assertPathExists('.bzr/README')
    self.assertFileEqual(b'hello', '.bzr/README')
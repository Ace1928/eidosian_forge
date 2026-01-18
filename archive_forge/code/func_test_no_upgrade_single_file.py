import os
from breezy import branch, osutils, tests, workingtree
from breezy.bzr import bzrdir
from breezy.tests.script import ScriptRunner
def test_no_upgrade_single_file(self):
    """There's one basis revision per tree.

        Since you can't actually change the basis for a single file at the
        moment, we don't let you think you can.

        See bug 557886.
        """
    self.make_branch_and_tree('.')
    self.build_tree_contents([('a/',), ('a/file', b'content')])
    sr = ScriptRunner()
    sr.run_script(self, '\n            $ brz update ./a\n            2>brz: ERROR: brz update can only update a whole tree, not a file or subdirectory\n            $ brz update ./a/file\n            2>brz: ERROR: brz update can only update a whole tree, not a file or subdirectory\n            $ brz update .\n            2>Tree is up to date at revision 0 of branch ...\n            $ cd a\n            $ brz update .\n            2>brz: ERROR: brz update can only update a whole tree, not a file or subdirectory\n            # however, you can update the whole tree from a subdirectory\n            $ brz update\n            2>Tree is up to date at revision 0 of branch ...\n            ')
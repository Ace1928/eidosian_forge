import os
import breezy.branch
from breezy import osutils, workingtree
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import (CaseInsensitiveFilesystemFeature,
def test_dupe_move(self):
    self.script_runner = script.ScriptRunner()
    self.script_runner.run_script(self, '\n        $ brz init brz-bug\n        Created a standalone tree (format: 2a)\n        $ cd brz-bug\n        $ mkdir dir\n        $ brz add\n        adding dir\n        $ echo text >> dir/test.txt\n        $ brz add\n        adding dir/test.txt\n        $ brz ci -m "Add files"\n        2>Committing to: .../brz-bug/\n        2>added dir\n        2>added dir/test.txt\n        2>Committed revision 1.\n        $ mv dir dir2\n        $ mv dir2/test.txt dir2/test2.txt\n        $ brz st\n        removed:\n          dir/\n          dir/test.txt\n        unknown:\n          dir2/\n        $ brz mv dir dir2\n        dir => dir2\n        $ brz st\n        removed:\n          dir/test.txt\n        renamed:\n          dir/ => dir2/\n        unknown:\n          dir2/test2.txt\n        $ brz mv dir/test.txt dir2/test2.txt\n        dir/test.txt => dir2/test2.txt\n        ')
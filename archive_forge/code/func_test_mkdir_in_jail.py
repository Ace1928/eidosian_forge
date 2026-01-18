from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
def test_mkdir_in_jail(self):
    self.run_script('\n$ mkdir dir\n$ cd dir\n$ mkdir ../dir2\n$ cd ..\n')
    self.assertPathExists('dir')
    self.assertPathExists('dir2')
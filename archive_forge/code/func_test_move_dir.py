from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
def test_move_dir(self):
    self.run_script('\n$ mkdir dir\n$ echo content >dir/file\n')
    self.run_script('$ mv dir new_name')
    self.assertPathDoesNotExist('dir')
    self.assertPathExists('new_name')
    self.assertPathExists('new_name/file')
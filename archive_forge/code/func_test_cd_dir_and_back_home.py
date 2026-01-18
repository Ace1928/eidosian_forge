from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
def test_cd_dir_and_back_home(self):
    self.assertEqual(self.test_dir, osutils.getcwd())
    self.run_script('\n$ mkdir dir\n$ cd dir\n')
    self.assertEqual(osutils.pathjoin(self.test_dir, 'dir'), osutils.getcwd())
    self.run_script('$ cd')
    self.assertEqual(self.test_dir, osutils.getcwd())
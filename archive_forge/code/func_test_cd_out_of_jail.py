from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
def test_cd_out_of_jail(self):
    self.assertRaises(ValueError, self.run_script, '$ cd /out-of-jail')
    self.assertRaises(ValueError, self.run_script, '$ cd ..')
from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
def test_cat_bogus_input_file(self):
    self.run_script('\n$ cat <file\n2>file: No such file or directory\n')
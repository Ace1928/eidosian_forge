from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
def test_dont_shelve(self):
    self.run_script("$ brz shelve -m 'shelve bar'\n2>Shelve? ([y]es, [N]o, [f]inish, [q]uit): \n2>No changes to shelve.\n", null_output_matches_anything=True)
    self.run_script('\n            $ brz st\n            modified:\n              file\n            ')
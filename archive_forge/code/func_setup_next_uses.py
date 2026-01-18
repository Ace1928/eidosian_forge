from typing import List
from breezy import branch, urlutils
from breezy.tests import script
def setup_next_uses(self):
    self.do_command(*self.first_use_args)
    self.run_script("\n            $ brz branch parent new_parent\n            $ cd new_parent\n            $ echo new parent > file\n            $ brz commit -m 'new parent'\n            $ cd ..\n            ", null_output_matches_anything=True)
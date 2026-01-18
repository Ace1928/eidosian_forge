from typing import List
from breezy import branch, urlutils
from breezy.tests import script
def test_next_uses_no_option(self):
    self.setup_next_uses()
    self.do_command(*self.next_uses_args)
    self.assertLocations(self.first_use_args)
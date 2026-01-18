from typing import List
from breezy import branch, urlutils
from breezy.tests import script
def test_next_uses_remember(self):
    self.setup_next_uses()
    self.do_command('--remember', *self.next_uses_args)
    self.assertLocations(self.next_uses_args)
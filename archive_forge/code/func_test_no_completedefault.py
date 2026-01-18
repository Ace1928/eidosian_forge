import cmd2
from cliff.interactive import InteractiveApp
from cliff.tests import base
def test_no_completedefault(self):
    self._test_completedefault([], 'taz ', 4)
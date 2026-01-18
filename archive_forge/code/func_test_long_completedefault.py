import cmd2
from cliff.interactive import InteractiveApp
from cliff.tests import base
def test_long_completedefault(self):
    self._test_completedefault(['long'], 'show  ', 6)
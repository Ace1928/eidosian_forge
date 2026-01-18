import inspect
import warnings
import debtcollector
from debtcollector.fixtures import disable
from debtcollector import moves
from debtcollector import removals
from debtcollector import renames
from debtcollector.tests import base as test_base
from debtcollector import updating
def test_keeps_argspec(self):
    self.assertEqual(inspect.getfullargspec(KittyKat.supermeow), inspect.getfullargspec(KittyKat.meow))
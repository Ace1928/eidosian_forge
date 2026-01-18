import inspect
import warnings
import debtcollector
from debtcollector.fixtures import disable
from debtcollector import moves
from debtcollector import removals
from debtcollector import renames
from debtcollector.tests import base as test_base
from debtcollector import updating
def test_argspec_preserved(self):
    self.assertEqual(inspect.getfullargspec(blip_blop_blip_unwrapped), inspect.getfullargspec(blip_blop_blip))
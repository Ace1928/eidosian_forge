import inspect
import warnings
import debtcollector
from debtcollector.fixtures import disable
from debtcollector import moves
from debtcollector import removals
from debtcollector import renames
from debtcollector.tests import base as test_base
from debtcollector import updating
@removals.remove(category=PendingDeprecationWarning)
def crimson_lightning_to_remove(fake_input=None):
    return fake_input
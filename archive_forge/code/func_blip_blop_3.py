import inspect
import warnings
import debtcollector
from debtcollector.fixtures import disable
from debtcollector import moves
from debtcollector import removals
from debtcollector import renames
from debtcollector.tests import base as test_base
from debtcollector import updating
@renames.renamed_kwarg('blip', 'blop', replace=True)
def blip_blop_3(blop=1):
    return blop
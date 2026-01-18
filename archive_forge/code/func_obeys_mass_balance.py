import math
from collections import OrderedDict, defaultdict
from itertools import chain
from .chemistry import Reaction, Substance
from .units import to_unitless
from .util.pyutil import deprecated
def obeys_mass_balance(self):
    """Returns True if all reactions obeys mass balance, else False."""
    for rxn in self.rxns:
        if rxn.mass_balance_violation(self.substances) != 0:
            return False
    return True
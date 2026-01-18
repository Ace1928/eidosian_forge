import math
from collections import OrderedDict, defaultdict
from itertools import chain
from .chemistry import Reaction, Substance
from .units import to_unitless
from .util.pyutil import deprecated
def obeys_charge_neutrality(self):
    """Returns False if any reaction violate charge neutrality."""
    for rxn in self.rxns:
        if rxn.charge_neutrality_violation(self.substances) != 0:
            return False
    return True
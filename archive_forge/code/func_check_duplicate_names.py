import math
from collections import OrderedDict, defaultdict
from itertools import chain
from .chemistry import Reaction, Substance
from .units import to_unitless
from .util.pyutil import deprecated
def check_duplicate_names(self, throw=False):
    names_seen = {}
    for idx, rxn in enumerate(self.rxns):
        if rxn.name is None:
            continue
        if rxn.name in names_seen:
            if throw:
                raise ValueError('Duplicate names at %d: %s' % (idx, rxn.name))
            else:
                return False
        else:
            names_seen[rxn.name] = idx
    return True
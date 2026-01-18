import math
from collections import OrderedDict, defaultdict
from itertools import chain
from .chemistry import Reaction, Substance
from .units import to_unitless
from .util.pyutil import deprecated
def identify_equilibria(self):
    """Returns a list of index pairs of reactions forming equilibria.

        The pairs are sorted with respect to index (lowest first)
        """
    eq = []
    for ri1, rxn1 in enumerate(self.rxns):
        for ri2, rxn2 in enumerate(self.rxns[ri1 + 1:], ri1 + 1):
            all_eq = rxn1.all_reac_stoich(self.substances) == rxn2.all_prod_stoich(self.substances) and rxn1.all_prod_stoich(self.substances) == rxn2.all_reac_stoich(self.substances)
            if all_eq:
                eq.append((ri1, ri2))
                break
    return eq
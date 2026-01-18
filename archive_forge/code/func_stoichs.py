import math
from collections import OrderedDict, defaultdict
from itertools import chain
from .chemistry import Reaction, Substance
from .units import to_unitless
from .util.pyutil import deprecated
def stoichs(self, non_precip_rids=()):
    """Conditional stoichiometries depending on precipitation status"""
    import numpy as np
    return np.array([-np.array(eq.precipitate_stoich(self.substances)[0]) if idx in non_precip_rids else eq.non_precipitate_stoich(self.substances) for idx, eq in enumerate(self.rxns)], dtype=object)
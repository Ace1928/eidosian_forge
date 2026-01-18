from math import tanh, sqrt, exp
from operator import itemgetter
import numpy as np
from ase.db.core import now
from ase.ga import get_raw_score
def get_current_population(self):
    """ Returns a copy of the current population. """
    self.update()
    return [a.copy() for a in self.pop]
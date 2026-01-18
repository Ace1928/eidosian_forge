from math import tanh, sqrt, exp
from operator import itemgetter
import numpy as np
from ase.db.core import now
from ase.ga import get_raw_score
def get_two_candidates(self):
    """ Returns two candidates for pairing employing the
            roulete wheel selection scheme described in
            R.L. Johnston Dalton Transactions,
            Vol. 22, No. 22. (2003), pp. 4193-4207
        """
    if len(self.pop) < 2:
        self.update()
    if len(self.pop) < 2:
        return None
    fit = self.current_fitness
    fmax = max(fit)
    c1 = self.pop[0]
    c2 = self.pop[0]
    while c1.info['confid'] == c2.info['confid']:
        nnf = True
        while nnf:
            t = self.rng.randint(len(self.pop))
            if fit[t] > self.rng.rand() * fmax:
                c1 = self.pop[t]
                nnf = False
        nnf = True
        while nnf:
            t = self.rng.randint(len(self.pop))
            if fit[t] > self.rng.rand() * fmax:
                c2 = self.pop[t]
                nnf = False
    return (c1.copy(), c2.copy())
import sys
import numpy as np
from itertools import combinations_with_replacement
import ase.units as u
from ase.parallel import parprint, paropen
from ase.vibrations.resonant_raman import ResonantRaman
from ase.vibrations.franck_condon import FranckCondonOverlap
from ase.vibrations.franck_condon import FranckCondonRecursive
def init_parallel_excitations(self):
    """Init for paralellization over excitations."""
    n_p = len(self.ex0E_p)
    exF_pr = self._collect_r(self.exF_rp, [n_p], self.ex0E_p.dtype).T
    myn = -(-n_p // self.comm.size)
    rank = self.comm.rank
    s = slice(myn * rank, myn * (rank + 1))
    return (n_p, range(n_p)[s], exF_pr)
import numpy as np
from numpy import linalg
from ase.transport.selfenergy import LeadSelfEnergy, BoxProbe
from ase.transport.greenfunction import GreenFunction
from ase.transport.tools import subdiagonalize, cutcoupling, dagger,\
from ase.units import kB
def cutcoupling_bfs(self, bfs, apply=False):
    self.initialize()
    bfs = np.array(bfs)
    p = self.input_parameters
    h_pp = p['h'].copy()
    s_pp = p['s'].copy()
    cutcoupling(h_pp, s_pp, bfs)
    if apply:
        self.uptodate = False
        p['h'][:] = h_pp
        p['s'][:] = s_pp
        for alpha, sigma in enumerate(self.selfenergies):
            for m in bfs:
                sigma.h_im[:, m] = 0.0
                sigma.s_im[:, m] = 0.0
    return (h_pp, s_pp)
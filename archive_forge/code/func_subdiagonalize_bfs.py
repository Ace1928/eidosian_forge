import numpy as np
from numpy import linalg
from ase.transport.selfenergy import LeadSelfEnergy, BoxProbe
from ase.transport.greenfunction import GreenFunction
from ase.transport.tools import subdiagonalize, cutcoupling, dagger,\
from ase.units import kB
def subdiagonalize_bfs(self, bfs, apply=False):
    self.initialize()
    bfs = np.array(bfs)
    p = self.input_parameters
    h_mm = p['h']
    s_mm = p['s']
    ht_mm, st_mm, c_mm, e_m = subdiagonalize(h_mm, s_mm, bfs)
    if apply:
        self.uptodate = False
        h_mm[:] = ht_mm.real
        s_mm[:] = st_mm.real
        for alpha, sigma in enumerate(self.selfenergies):
            sigma.h_im[:] = np.dot(sigma.h_im, c_mm)
            sigma.s_im[:] = np.dot(sigma.s_im, c_mm)
    c_mm = np.take(c_mm, bfs, axis=0)
    c_mm = np.take(c_mm, bfs, axis=1)
    return (ht_mm, st_mm, e_m.real, c_mm)
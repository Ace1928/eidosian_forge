import numpy as np
from numpy import linalg
from ase.transport.selfenergy import LeadSelfEnergy, BoxProbe
from ase.transport.greenfunction import GreenFunction
from ase.transport.tools import subdiagonalize, cutcoupling, dagger,\
from ase.units import kB
def print_pl_convergence(self):
    self.initialize()
    pl1 = len(self.input_parameters['h1']) // 2
    h_ii = self.selfenergies[0].h_ii
    s_ii = self.selfenergies[0].s_ii
    ha_ii = self.greenfunction.H[:pl1, :pl1]
    sa_ii = self.greenfunction.S[:pl1, :pl1]
    c1 = np.abs(h_ii - ha_ii).max()
    c2 = np.abs(s_ii - sa_ii).max()
    print('Conv (h,s)=%.2e, %2.e' % (c1, c2))
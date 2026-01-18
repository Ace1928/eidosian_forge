import numpy as np
from numpy import linalg
from ase.transport.selfenergy import LeadSelfEnergy, BoxProbe
from ase.transport.greenfunction import GreenFunction
from ase.transport.tools import subdiagonalize, cutcoupling, dagger,\
from ase.units import kB
def lowdin_rotation(self, apply=False):
    p = self.input_parameters
    h_mm = p['h']
    s_mm = p['s']
    eig, rot_mm = linalg.eigh(s_mm)
    eig = np.abs(eig)
    rot_mm = np.dot(rot_mm / np.sqrt(eig), dagger(rot_mm))
    if apply:
        self.uptodate = False
        h_mm[:] = rotate_matrix(h_mm, rot_mm)
        s_mm[:] = rotate_matrix(s_mm, rot_mm)
        for alpha, sigma in enumerate(self.selfenergies):
            sigma.h_im[:] = np.dot(sigma.h_im, rot_mm)
            sigma.s_im[:] = np.dot(sigma.s_im, rot_mm)
    return rot_mm
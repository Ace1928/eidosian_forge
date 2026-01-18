import time
import numpy as np
from ase.transport.tools import dagger
from ase.transport.selfenergy import LeadSelfEnergy
from ase.transport.greenfunction import GreenFunction
from ase.parallel import world
def get_transmission(self, v_12, v_11_2=None, v_22_1=None):
    """XXX

        v_12:
            coupling between tip and surface 
        v_11_2:
            correction to "on-site" tip elements due to the 
            surface (eq.16). Is only included to first order.
        v_22_1:
            corretion to "on-site" surface elements due to he
            tip (eq.17). Is only included to first order.
        """
    dim0 = v_12.shape[0]
    dim1 = v_12.shape[1]
    nenergies = len(self.energies)
    T_e = np.empty(nenergies, float)
    v_21 = dagger(v_12)
    for e, energy in enumerate(self.energies):
        gft1 = self.gft1_emm[e]
        if v_11_2 != None:
            gf1 = np.dot(v_11_2, np.dot(gft1, v_11_2))
            gf1 += gft1
        else:
            gf1 = gft1
        gft2 = self.gft2_emm[e]
        if v_22_1 != None:
            gf2 = np.dot(v_22_1, np.dot(gft2, v_22_1))
            gf2 += gft2
        else:
            gf2 = gft2
        a1 = gf1 - dagger(gf1)
        a2 = gf2 - dagger(gf2)
        self.v_12 = v_12
        self.a2 = a2
        self.v_21 = v_21
        self.a1 = a1
        v12_a2 = np.dot(v_12, a2[:dim1])
        v21_a1 = np.dot(v_21, a1[-dim0:])
        self.v12_a2 = v12_a2
        self.v21_a1 = v21_a1
        T = -np.trace(np.dot(v12_a2[:, :dim1], v21_a1[:, -dim0:]))
        assert abs(T.imag).max() < 1e-14
        T_e[e] = T.real
        self.T_e = T_e
    return T_e
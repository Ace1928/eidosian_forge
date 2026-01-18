import numpy as np
from scipy.optimize import fsolve
from scipy.sparse import coo_matrix
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxModel
def reactor_outlet_concentrations(sv, caf, k1, k2, k3):

    def _model(x, sv, caf, k1, k2, k3):
        ca, cb, cc, cd = (x[0], x[1], x[2], x[3])
        r = np.zeros(4)
        r[0] = sv * caf + (-sv - k1) * ca - 2 * k3 * ca ** 2
        r[1] = k1 * ca + (-sv - k2) * cb
        r[2] = k2 * cb - sv * cc
        r[3] = k3 * ca ** 2 - sv * cd
        return r
    concentrations = fsolve(lambda x: _model(x, sv, caf, k1, k2, k3), np.ones(4), xtol=1e-08)
    return concentrations
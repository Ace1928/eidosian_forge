import numpy as np
from numpy import linalg
from ase.transport.selfenergy import LeadSelfEnergy, BoxProbe
from ase.transport.greenfunction import GreenFunction
from ase.transport.tools import subdiagonalize, cutcoupling, dagger,\
from ase.units import kB
def get_left_channels(self, energy, nchan=1):
    self.initialize()
    g_s_ii = self.greenfunction.retarded(energy)
    lambda_l_ii = self.selfenergies[0].get_lambda(energy)
    lambda_r_ii = self.selfenergies[1].get_lambda(energy)
    if self.greenfunction.S is not None:
        s_mm = self.greenfunction.S
        s_s_i, s_s_ii = linalg.eig(s_mm)
        s_s_i = np.abs(s_s_i)
        s_s_sqrt_i = np.sqrt(s_s_i)
        s_s_sqrt_ii = np.dot(s_s_ii * s_s_sqrt_i, dagger(s_s_ii))
        s_s_isqrt_ii = np.dot(s_s_ii / s_s_sqrt_i, dagger(s_s_ii))
    lambdab_r_ii = np.dot(np.dot(s_s_isqrt_ii, lambda_r_ii), s_s_isqrt_ii)
    a_l_ii = np.dot(np.dot(g_s_ii, lambda_l_ii), dagger(g_s_ii))
    ab_l_ii = np.dot(np.dot(s_s_sqrt_ii, a_l_ii), s_s_sqrt_ii)
    lambda_i, u_ii = linalg.eig(ab_l_ii)
    ut_ii = np.sqrt(lambda_i / (2.0 * np.pi)) * u_ii
    m_ii = 2 * np.pi * np.dot(np.dot(dagger(ut_ii), lambdab_r_ii), ut_ii)
    T_i, c_in = linalg.eig(m_ii)
    T_i = np.abs(T_i)
    channels = np.argsort(-T_i)[:nchan]
    c_in = np.take(c_in, channels, axis=1)
    T_n = np.take(T_i, channels)
    v_in = np.dot(np.dot(s_s_isqrt_ii, ut_ii), c_in)
    return (T_n, v_in)
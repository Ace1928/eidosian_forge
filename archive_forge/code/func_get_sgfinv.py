import numpy as np
def get_sgfinv(self, energy):
    """The inverse of the retarded surface Green function"""
    z = energy - self.bias + self.eta * 1j
    v_00 = z * self.s_ii.T.conj() - self.h_ii.T.conj()
    v_11 = v_00.copy()
    v_10 = z * self.s_ij - self.h_ij
    v_01 = z * self.s_ij.T.conj() - self.h_ij.T.conj()
    delta = self.conv + 1
    while delta > self.conv:
        a = np.linalg.solve(v_11, v_01)
        b = np.linalg.solve(v_11, v_10)
        v_01_dot_b = np.dot(v_01, b)
        v_00 -= v_01_dot_b
        v_11 -= np.dot(v_10, a)
        v_11 -= v_01_dot_b
        v_01 = -np.dot(v_01, a)
        v_10 = -np.dot(v_10, b)
        delta = abs(v_01).max()
    return v_00
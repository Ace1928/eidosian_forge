import numpy as np
from ase.atoms import Atoms
def euler_angles(self, mode='zyz'):
    """Return three Euler angles describing the rotation, in radians.
        Mode can be zyz or zxz. Default is zyz."""
    if mode == 'zyz':
        apc = np.arctan2(self.q[3], self.q[0])
        amc = np.arctan2(self.q[1], self.q[2])
        a, c = (apc + amc, apc - amc)
        cos_amc = np.cos(amc)
        if cos_amc != 0:
            sinb2 = self.q[2] / cos_amc
        else:
            sinb2 = self.q[1] / np.sin(amc)
        cos_apc = np.cos(apc)
        if cos_apc != 0:
            cosb2 = self.q[0] / cos_apc
        else:
            cosb2 = self.q[3] / np.sin(apc)
        b = np.arctan2(sinb2, cosb2) * 2
    elif mode == 'zxz':
        apc = np.arctan2(self.q[3], self.q[0])
        amc = np.arctan2(-self.q[2], self.q[1])
        a, c = (apc + amc, apc - amc)
        cos_amc = np.cos(amc)
        if cos_amc != 0:
            sinb2 = self.q[1] / cos_amc
        else:
            sinb2 = self.q[2] / np.sin(amc)
        cos_apc = np.cos(apc)
        if cos_apc != 0:
            cosb2 = self.q[0] / cos_apc
        else:
            cosb2 = self.q[3] / np.sin(apc)
        b = np.arctan2(sinb2, cosb2) * 2
    else:
        raise ValueError('Invalid Euler angles mode {0}'.format(mode))
    return np.array([a, b, c])
import pytest
import numpy as np
from ase.quaternions import Quaternion
def eulang_rotm(a, b, c, mode='zyz'):
    rota = axang_rotm([0, 0, 1], a)
    rotc = axang_rotm([0, 0, 1], c)
    if mode == 'zyz':
        rotb = axang_rotm([0, 1, 0], b)
    elif mode == 'zxz':
        rotb = axang_rotm([1, 0, 0], b)
    return np.dot(rotc, np.dot(rotb, rota))
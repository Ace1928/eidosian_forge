import pytest
import numpy as np
from ase.quaternions import Quaternion
def axang_rotm(u, theta):
    u = np.array(u, float)
    u /= np.linalg.norm(u)
    ucpm = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    rotm = np.cos(theta) * np.identity(3) + np.sin(theta) * ucpm + (1 - np.cos(theta)) * np.kron(u[:, None], u[None, :])
    return rotm
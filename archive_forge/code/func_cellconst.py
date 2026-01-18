import numpy as np
from ase import Atoms
from ase.units import Bohr, Ry
from ase.utils import reader, writer
def cellconst(metT):
    """ metT=np.dot(cell,cell.T) """
    aa = np.sqrt(metT[0, 0])
    bb = np.sqrt(metT[1, 1])
    cc = np.sqrt(metT[2, 2])
    gamma = np.arccos(metT[0, 1] / (aa * bb)) / np.pi * 180.0
    beta = np.arccos(metT[0, 2] / (aa * cc)) / np.pi * 180.0
    alpha = np.arccos(metT[1, 2] / (bb * cc)) / np.pi * 180.0
    return np.array([aa, bb, cc, alpha, beta, gamma])
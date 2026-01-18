import time
import unittest
import numpy
from scipy.optimize import linear_sum_assignment
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdmolops
from rdkit.Chem.Fingerprints import FingerprintMols
def getsimab(mappings, simmatrixdict):
    """return the similarity for a set of mapping.  See Eqn 3"""
    naa, nab = simmatrixdict.shape
    score = 0.0
    for a, b in mappings:
        score += simmatrixdict[a][b]
    simab = score / (max(naa, nab) * 2 - score)
    return simab
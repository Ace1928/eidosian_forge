import time
import unittest
import numpy
from scipy.optimize import linear_sum_assignment
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdmolops
from rdkit.Chem.Fingerprints import FingerprintMols
def gethungarianmappings(simmatrixarray):
    """return a mapping of the atoms in the similarity matrix - the Hungarian algorithm is used because it is invariant to atom ordering.  Requires scipy"""
    costarray = numpy.ones(simmatrixarray.shape) - simmatrixarray
    row_ind, col_ind = linear_sum_assignment(costarray)
    res = zip(row_ind, col_ind)
    return res
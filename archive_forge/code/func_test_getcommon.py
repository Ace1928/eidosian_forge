import time
import unittest
import numpy
from scipy.optimize import linear_sum_assignment
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdmolops
from rdkit.Chem.Fingerprints import FingerprintMols
def test_getcommon(self):
    self.assertEqual(getcommon([2, 2, 2, 3, 3, 3], 6, [1, 2, 3, 3, 4, 5], 6), 3)
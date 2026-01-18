import re
from collections import Counter, defaultdict, namedtuple
import numpy as np
import seaborn as sns
from numpy import linalg
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
def calcSP3CarbonSubstituentMoment(subSizes):
    if len(subSizes) != 4:
        raise ValueError('Function "calcSP3CarbonSubstituentMoment" expects an array of size 4 as parameter')
    x1 = np.array([1, 1, 1])
    x2 = np.array([-1, 1, -1])
    x3 = np.array([1, -1, -1])
    x4 = np.array([-1, -1, 1])
    substituentMoment = linalg.norm(subSizes[0] * x1 + subSizes[1] * x2 + subSizes[2] * x3 + subSizes[3] * x4)
    return substituentMoment
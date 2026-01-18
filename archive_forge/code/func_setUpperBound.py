import numpy
from rdkit import Geometry
from rdkit.Chem import ChemicalFeatures
from rdkit.RDLogger import logger
def setUpperBound(self, i, j, val, checkBounds=False):
    if checkBounds:
        self._checkBounds(i, j)
    if i > j:
        j, i = (i, j)
    self._boundsMat[i, j] = val
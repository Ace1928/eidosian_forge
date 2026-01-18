import numpy
from rdkit import Geometry
from rdkit.Chem import ChemicalFeatures
from rdkit.RDLogger import logger
def getUpperBound(self, i, j):
    if i > j:
        j, i = (i, j)
    return self._boundsMat[i, j]
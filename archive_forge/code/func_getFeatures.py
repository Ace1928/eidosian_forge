import numpy
from rdkit import Geometry
from rdkit.Chem import ChemicalFeatures
from rdkit.RDLogger import logger
def getFeatures(self):
    return self._feats
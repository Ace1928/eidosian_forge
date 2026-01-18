import numpy
from rdkit import Geometry
from rdkit.Chem import ChemicalFeatures
from rdkit.RDLogger import logger
def initFromFile(self, inF):
    self.initFromLines(inF.readlines())
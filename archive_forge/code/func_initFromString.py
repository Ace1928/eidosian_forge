import numpy
from rdkit import Geometry
from rdkit.Chem import ChemicalFeatures
from rdkit.RDLogger import logger
def initFromString(self, text):
    lines = text.split('\\n')
    self.initFromLines(lines)
import re
from collections import Counter, defaultdict, namedtuple
import numpy as np
import seaborn as sns
from numpy import linalg
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
def getNumAromaticBondsSubstituent(mol, subAtoms):
    return sum((1 for b in getBondsSubstituent(mol, subAtoms) if mol.GetBondWithIdx(b).GetIsAromatic()))
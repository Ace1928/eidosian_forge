import re
import sys
from optparse import OptionParser
from rdkit import Chem
def get_symmetry_class(smi):
    symmetry = []
    m = Chem.MolFromSmiles(smi)
    symmetry_classes = Chem.CanonicalRankAtoms(m, breakTies=False)
    for atom, symmetry_class in zip(m.GetAtoms(), symmetry_classes):
        if atom.GetMass() == 0:
            symmetry.append(symmetry_class)
    return symmetry
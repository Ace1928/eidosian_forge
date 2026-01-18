from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem.ChemUtils.DescriptorUtilities import setDescriptorVersion
from collections import defaultdict
def calculateSpacialScore(self):
    """Calculates the total spacial score for a molecule"""
    score = 0
    for atom in self.mol.GetAtoms():
        atom_idx = atom.GetIdx()
        self.hyb_score[atom_idx] = self._accountForHybridisation(atom)
        self.stereo_score[atom_idx] = self._accountForStereo(atom_idx)
        self.ring_score[atom_idx] = self._accountForRing(atom)
        self.bond_score[atom_idx] = self._accountForNeighbors(atom)
        score += self._calculateScoreForAtom(atom_idx)
    return score
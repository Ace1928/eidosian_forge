from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem.ChemUtils.DescriptorUtilities import setDescriptorVersion
from collections import defaultdict
def findStereoAtomIdxs(self, includeUnassigned=True):
    """Finds indices of atoms that are (pseudo)stereo/chiralcentres, in respect to the attached groups (does not account for double bond isomers)"""
    stereo_centers = Chem.FindMolChiralCenters(self.mol, includeUnassigned=includeUnassigned, includeCIP=False, useLegacyImplementation=False)
    stereo_idxs = [atom_idx for atom_idx, _ in stereo_centers]
    return stereo_idxs
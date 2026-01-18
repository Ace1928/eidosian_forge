from warnings import warn
import copy
import logging
from rdkit import Chem
from rdkit.Chem.rdchem import BondDir, BondStereo, BondType
from .utils import memoized_property, pairwise
@memoized_property
def smarts(self):
    return Chem.MolFromSmarts(self.smarts_str)
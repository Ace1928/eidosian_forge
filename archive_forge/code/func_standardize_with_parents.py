from warnings import warn
import copy
import logging
from rdkit import Chem
from .charge import ACID_BASE_PAIRS, CHARGE_CORRECTIONS, Reionizer, Uncharger
from .fragment import PREFER_ORGANIC, FragmentRemover, LargestFragmentChooser
from .metal import MetalDisconnector
from .normalize import MAX_RESTARTS, NORMALIZATIONS, Normalizer
from .tautomer import (MAX_TAUTOMERS, TAUTOMER_SCORES, TAUTOMER_TRANSFORMS, TautomerCanonicalizer,
from .utils import memoized_property
def standardize_with_parents(self, mol):
    """"""
    standardized = self.standardize(mol)
    tautomer = self.tautomer_parent(standardized, skip_standardize=True)
    super = self.super_parent(standardized, skip_standardize=True)
    mols = {'standardized': standardized, 'tautomer_parent': tautomer, 'super_parent': super}
    return mols
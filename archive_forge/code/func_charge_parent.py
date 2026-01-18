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
def charge_parent(self, mol, skip_standardize=False):
    """Return the charge parent of a given molecule.

        The charge parent is the uncharged version of the fragment parent.

        :param mol: The input molecule.
        :type mol: :rdkit:`Mol <Chem.rdchem.Mol-class.html>`
        :param bool skip_standardize: Set to True if mol has already been standardized.
        :returns: The charge parent molecule.
        :rtype: :rdkit:`Mol <Chem.rdchem.Mol-class.html>`
        """
    if not skip_standardize:
        mol = self.standardize(mol)
    fragment = self.fragment_parent(mol, skip_standardize=True)
    if fragment:
        uncharged = self.uncharge(fragment)
        uncharged = self.standardize(uncharged)
        return uncharged
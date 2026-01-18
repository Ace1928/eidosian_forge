from warnings import warn
import logging
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from .utils import memoized_property
def is_organic(fragment):
    """Return true if fragment contains at least one carbon atom.

    :param fragment: The fragment as an RDKit Mol object.
    """
    return any((atom.GetAtomicNum() == 6 for atom in fragment.GetAtoms()))
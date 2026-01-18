import dataclasses
import re
import string
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple
import numpy as np
from . import residue_constants
def ideal_atom_mask(prot: Protein) -> np.ndarray:
    """Computes an ideal atom mask.

    `Protein.atom_mask` typically is defined according to the atoms that are reported in the PDB. This function
    computes a mask according to heavy atoms that should be present in the given sequence of amino acids.

    Args:
      prot: `Protein` whose fields are `numpy.ndarray` objects.

    Returns:
      An ideal atom mask.
    """
    return residue_constants.STANDARD_ATOM_MASK[prot.aatype]
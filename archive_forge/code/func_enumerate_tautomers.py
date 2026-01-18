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
@memoized_property
def enumerate_tautomers(self):
    """
        :returns: A callable :class:`~molvs.tautomer.TautomerEnumerator` instance.
        """
    return TautomerEnumerator(transforms=self.tautomer_transforms, max_tautomers=self.max_tautomers)
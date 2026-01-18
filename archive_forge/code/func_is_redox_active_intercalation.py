from __future__ import annotations
import math
from collections import defaultdict
import scipy.constants as const
from pymatgen.core import Composition, Element, Species
def is_redox_active_intercalation(element) -> bool:
    """True if element is redox active and interesting for intercalation materials.

    Args:
        element: Element object
    """
    ns = ['Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Nb', 'Mo', 'W', 'Sb', 'Sn', 'Bi']
    return element.symbol in ns
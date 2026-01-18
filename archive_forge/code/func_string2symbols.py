from typing import List, Sequence, Set, Dict, Union, Iterator
import warnings
import collections.abc
import numpy as np
from ase.data import atomic_numbers, chemical_symbols
from ase.formula import Formula
def string2symbols(s: str) -> List[str]:
    """Convert string to list of chemical symbols."""
    return list(Formula(s))
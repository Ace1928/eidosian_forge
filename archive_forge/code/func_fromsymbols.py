from typing import List, Sequence, Set, Dict, Union, Iterator
import warnings
import collections.abc
import numpy as np
from ase.data import atomic_numbers, chemical_symbols
from ase.formula import Formula
@classmethod
def fromsymbols(cls, symbols) -> 'Symbols':
    numbers = symbols2numbers(symbols)
    return cls(np.array(numbers))
from typing import List, Sequence, Set, Dict, Union, Iterator
import warnings
import collections.abc
import numpy as np
from ase.data import atomic_numbers, chemical_symbols
from ase.formula import Formula
def symbols2numbers(symbols) -> List[int]:
    if isinstance(symbols, str):
        symbols = string2symbols(symbols)
    numbers = []
    for s in symbols:
        if isinstance(s, str):
            numbers.append(atomic_numbers[s])
        else:
            numbers.append(int(s))
    return numbers
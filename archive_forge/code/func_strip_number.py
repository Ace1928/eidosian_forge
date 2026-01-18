from math import gcd
import re
from typing import Dict, Tuple, List, Sequence, Union
from ase.data import chemical_symbols, atomic_numbers
def strip_number(s: str) -> Tuple[int, str]:
    m = re.match('[0-9]*', s)
    assert m is not None
    return (int(m.group() or 1), s[m.end():])
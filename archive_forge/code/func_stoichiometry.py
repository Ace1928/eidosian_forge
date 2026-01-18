from math import gcd
import re
from typing import Dict, Tuple, List, Sequence, Union
from ase.data import chemical_symbols, atomic_numbers
def stoichiometry(self) -> Tuple['Formula', 'Formula', int]:
    """Reduce to unique stoichiomerty using "chemical symbols" A, B, C, ...

        Examples
        --------
        >>> Formula('CO2').stoichiometry()
        (Formula('AB2'), Formula('CO2'), 1)
        >>> Formula('(H2O)4').stoichiometry()
        (Formula('AB2'), Formula('OH2'), 4)
        """
    count1, N = self._reduce()
    c = ord('A')
    count2 = {}
    count3 = {}
    for n, symb in sorted(((n, symb) for symb, n in count1.items())):
        count2[chr(c)] = n
        count3[symb] = n
        c += 1
    return (self.from_dict(count2), self.from_dict(count3), N)
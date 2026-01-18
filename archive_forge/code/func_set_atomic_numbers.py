from typing import List, Optional
import numpy as np
from ase.data import atomic_numbers as ref_atomic_numbers
from ase.spacegroup import Spacegroup
from ase.cluster.base import ClusterBase
from ase.cluster.cluster import Cluster
def set_atomic_numbers(self, symbols):
    """Extract atomic number from element"""
    atomic_numbers = []
    if self.element_basis is None:
        if isinstance(symbols, str):
            atomic_numbers.append(ref_atomic_numbers[symbols])
        elif isinstance(symbols, int):
            atomic_numbers.append(symbols)
        else:
            raise TypeError('The symbol argument must be a ' + 'string or an atomic number.')
        element_basis = [0] * len(self.atomic_basis)
    else:
        if isinstance(symbols, (list, tuple)):
            nsymbols = len(symbols)
        else:
            nsymbols = 0
        nelement_basis = max(self.element_basis) + 1
        if nsymbols != nelement_basis:
            raise TypeError('The symbol argument must be a sequence ' + 'of length %d' % (nelement_basis,) + ' (one for each kind of lattice position')
        for s in symbols:
            if isinstance(s, str):
                atomic_numbers.append(ref_atomic_numbers[s])
            elif isinstance(s, int):
                atomic_numbers.append(s)
            else:
                raise TypeError('The symbol argument must be a ' + 'string or an atomic number.')
        element_basis = self.element_basis
    self.atomic_numbers = [atomic_numbers[n] for n in element_basis]
    assert len(self.atomic_numbers) == len(self.atomic_basis)
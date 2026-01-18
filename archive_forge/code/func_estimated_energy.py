import collections
from typing import cast, Dict, Optional, Union, TYPE_CHECKING
import numpy as np
from cirq import ops
from cirq.work import collector
def estimated_energy(self) -> Union[float, complex]:
    """Sums up the sampled expectations, weighted by their coefficients."""
    energy = 0j
    for pauli_string, coef in self._pauli_coef_terms:
        a = self._zeros[pauli_string]
        b = self._ones[pauli_string]
        if a + b:
            energy += coef * (a - b) / (a + b)
    energy = complex(energy)
    energy += self._identity_offset
    if energy.imag == 0:
        energy = energy.real
    return energy
import itertools
from typing import Dict, Iterator, List, Optional, Sequence, cast
import numpy as np
@staticmethod
def from_bit_packed_string(coef_string: bytes) -> 'Wavefunction':
    """
        From a bit packed string, unpacks to get the wavefunction
        :param coef_string:
        """
    num_cfloat = len(coef_string) // OCTETS_PER_COMPLEX_DOUBLE
    amplitude_vector: np.ndarray = np.ndarray(shape=(num_cfloat,), buffer=coef_string, dtype='>c16')
    return Wavefunction(amplitude_vector)
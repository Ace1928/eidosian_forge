import itertools
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, TYPE_CHECKING
import numpy as np
from cirq import protocols, value
from cirq.linalg import transformations
from cirq.ops import raw_types, common_gates, pauli_gates, identity
@property
def error_probabilities(self) -> Dict[str, float]:
    """A dictionary from Pauli gates to probability"""
    return self._error_probabilities
import abc
from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
from cirq import protocols
from cirq._doc import document
def stabilized_by(self) -> Tuple[int, 'cirq.Pauli']:
    from cirq import ops
    return (self.eigenvalue, ops.Z)
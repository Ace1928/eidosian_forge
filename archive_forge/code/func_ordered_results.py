import collections
from typing import Dict, Counter, List, Optional, Sequence
import numpy as np
import cirq
def ordered_results(self, key: Optional[str]=None) -> List[int]:
    """Returns a list of arbitrarily but consistently ordered results as big endian ints.

        If a key parameter is supplied, these are the counts for the measurement results for
        the qubits measured by the measurement gate with that key.  If no key is given, these
        are the measurement results from measuring all qubits in the circuit.

        The value in the returned list is the computational basis state measured for the
        qubits that have been measured.  This is expressed in big-endian form. For example, if
        no measurement key is supplied and all qubits are measured, each entry in this returned dict
        has a bit string where the `cirq.LineQubit`s are expressed in the order:
            (cirq.LineQubit(0), cirq.LineQubit(1), ..., cirq.LineQubit(n-1))
        In the case where only `r` qubits are measured corresponding to targets t_0, t_1,...t_{r-1},
        the bit string corresponds to the order
            (cirq.LineQubit(t_0), cirq.LineQubit(t_1), ... cirq.LineQubit(t_{r-1}))
        """
    if key is not None and (not key in self._measurement_dict):
        raise ValueError(f'Measurement key {key} is not a key for a measurement gate in thecircuit that produced these results.')
    targets = self._measurement_dict[key] if key is not None else range(self.num_qubits())
    result: List[int] = []
    for value, count in self._counts.items():
        bits = [value >> self.num_qubits() - target - 1 & 1 for target in targets]
        bit_value = sum((bit * (1 << i) for i, bit in enumerate(bits[::-1])))
        result.extend([bit_value] * count)
    return result
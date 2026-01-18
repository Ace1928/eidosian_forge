from __future__ import annotations
from typing import Union, Callable, Optional, TYPE_CHECKING
from qiskit.circuit import QuantumCircuit
from qiskit.utils import optionals as _optionals
def evaluate_bitstring(self, bitstring: str) -> bool:
    """Evaluate the oracle on a bitstring.
        This evaluation is done classically without any quantum circuit.

        Args:
            bitstring: The bitstring for which to evaluate. The input bitstring is expected to be
                in little-endian order.

        Returns:
            True if the bitstring is a good state, False otherwise.
        """
    return self.boolean_expression.simulate(bitstring[::-1])
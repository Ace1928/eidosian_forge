from __future__ import annotations
import copy
from abc import abstractmethod
import numpy as np
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit.quantum_info.operators.op_shape import OpShape
from qiskit.quantum_info.operators.operator import Operator
from qiskit.result.counts import Counts
@abstractmethod
def probabilities(self, qargs: None | list=None, decimals: None | int=None) -> np.ndarray:
    """Return the subsystem measurement probability vector.

        Measurement probabilities are with respect to measurement in the
        computation (diagonal) basis.

        Args:
            qargs (None or list): subsystems to return probabilities for,
                if None return for all subsystems (Default: None).
            decimals (None or int): the number of decimal places to round
                values. If None no rounding is done (Default: None).

        Returns:
            np.array: The Numpy vector array of probabilities.
        """
    pass
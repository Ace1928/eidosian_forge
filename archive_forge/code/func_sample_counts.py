from __future__ import annotations
import copy
from abc import abstractmethod
import numpy as np
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit.quantum_info.operators.op_shape import OpShape
from qiskit.quantum_info.operators.operator import Operator
from qiskit.result.counts import Counts
def sample_counts(self, shots: int, qargs: None | list=None) -> Counts:
    """Sample a dict of qubit measurement outcomes in the computational basis.

        Args:
            shots (int): number of samples to generate.
            qargs (None or list): subsystems to sample measurements for,
                                if None sample measurement of all
                                subsystems (Default: None).

        Returns:
            Counts: sampled counts dictionary.

        Additional Information:

            This function *samples* measurement outcomes using the measure
            :meth:`probabilities` for the current state and `qargs`. It does
            not actually implement the measurement so the current state is
            not modified.

            The seed for random number generator used for sampling can be
            set to a fixed value by using the stats :meth:`seed` method.
        """
    samples = self.sample_memory(shots, qargs=qargs)
    inds, counts = np.unique(samples, return_counts=True)
    return Counts(zip(inds, counts))
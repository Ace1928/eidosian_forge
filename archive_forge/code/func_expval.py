from pathlib import Path
from typing import List, Sequence
from warnings import warn
import numpy as np
from pennylane_lightning.core.lightning_base import (
def expval(self, observable, shot_range=None, bin_size=None):
    """Expectation value of the supplied observable.

            Args:
                observable: A PennyLane observable.
                shot_range (tuple[int]): 2-tuple of integers specifying the range of samples
                    to use. If not specified, all samples are used.
                bin_size (int): Divides the shot range into bins of size ``bin_size``, and
                    returns the measurement statistic separately over each bin. If not
                    provided, the entire shot range is treated as a single bin.

            Returns:
                Expectation value of the observable
            """
    if observable.name in ['Projector']:
        diagonalizing_gates = observable.diagonalizing_gates()
        if self.shots is None and diagonalizing_gates:
            self.apply(diagonalizing_gates)
        results = super().expval(observable, shot_range=shot_range, bin_size=bin_size)
        if self.shots is None and diagonalizing_gates:
            self.apply([qml.adjoint(g, lazy=False) for g in reversed(diagonalizing_gates)])
        return results
    if self.shots is not None:
        samples = self.sample(observable, shot_range=shot_range, bin_size=bin_size)
        return np.squeeze(np.mean(samples, axis=0))
    measurements = MeasurementsC64(self.state_vector) if self.use_csingle else MeasurementsC128(self.state_vector)
    if observable.name == 'SparseHamiltonian':
        csr_hamiltonian = observable.sparse_matrix(wire_order=self.wires).tocsr(copy=False)
        return measurements.expval(csr_hamiltonian.indptr, csr_hamiltonian.indices, csr_hamiltonian.data)
    if observable.name in ['Hamiltonian', 'Hermitian'] or observable.arithmetic_depth > 0 or isinstance(observable.name, List):
        ob_serialized = QuantumScriptSerializer(self.short_name, self.use_csingle)._ob(observable, self.wire_map)
        return measurements.expval(ob_serialized)
    observable_wires = self.map_wires(observable.wires)
    return measurements.expval(observable.name, observable_wires)
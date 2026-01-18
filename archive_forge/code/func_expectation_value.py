from typing import Optional, List, Tuple, Iterable, Callable, Union, Dict
import numpy as np
from qiskit.exceptions import QiskitError
from ..distributions.quasi import QuasiDistribution
from ..counts import Counts
from .base_readout_mitigator import BaseReadoutMitigator
from .utils import counts_probability_vector, z_diagonal, str2diag
def expectation_value(self, data: Counts, diagonal: Union[Callable, dict, str, np.ndarray]=None, qubits: Iterable[int]=None, clbits: Optional[List[int]]=None, shots: Optional[int]=None) -> Tuple[float, float]:
    """Compute the mitigated expectation value of a diagonal observable.

        This computes the mitigated estimator of
        :math:`\\langle O \\rangle = \\mbox{Tr}[\\rho. O]` of a diagonal observable
        :math:`O = \\sum_{x\\in\\{0, 1\\}^n} O(x)|x\\rangle\\!\\langle x|`.

        Args:
            data: Counts object
            diagonal: Optional, the vector of diagonal values for summing the
                      expectation value. If ``None`` the default value is
                      :math:`[1, -1]^\\otimes n`.
            qubits: Optional, the measured physical qubits the count
                    bitstrings correspond to. If None qubits are assumed to be
                    :math:`[0, ..., n-1]`.
            clbits: Optional, if not None marginalize counts to the specified bits.
            shots: the number of shots.

        Returns:
            (float, float): the expectation value and an upper bound of the standard deviation.

        Additional Information:
            The diagonal observable :math:`O` is input using the ``diagonal`` kwarg as
            a list or Numpy array :math:`[O(0), ..., O(2^n -1)]`. If no diagonal is specified
            the diagonal of the Pauli operator
            :math`O = \\mbox{diag}(Z^{\\otimes n}) = [1, -1]^{\\otimes n}` is used.
            The ``clbits`` kwarg is used to marginalize the input counts dictionary
            over the specified bit-values, and the ``qubits`` kwarg is used to specify
            which physical qubits these bit-values correspond to as
            ``circuit.measure(qubits, clbits)``.
        """
    if qubits is None:
        qubits = self._qubits
    num_qubits = len(qubits)
    probs_vec, shots = counts_probability_vector(data, qubit_index=self._qubit_index, clbits=clbits, qubits=qubits)
    qubit_indices = [self._qubit_index[qubit] for qubit in qubits]
    ainvs = self._mitigation_mats[qubit_indices]
    if diagonal is None:
        diagonal = z_diagonal(2 ** num_qubits)
    elif isinstance(diagonal, str):
        diagonal = str2diag(diagonal)
    coeffs = np.reshape(diagonal, num_qubits * [2])
    einsum_args = [coeffs, list(range(num_qubits))]
    for i, ainv in enumerate(reversed(ainvs)):
        einsum_args += [ainv.T, [num_qubits + i, i]]
    einsum_args += [list(range(num_qubits, 2 * num_qubits))]
    coeffs = np.einsum(*einsum_args).ravel()
    expval = coeffs.dot(probs_vec)
    stddev_upper_bound = self.stddev_upper_bound(shots, qubits)
    return (expval, stddev_upper_bound)
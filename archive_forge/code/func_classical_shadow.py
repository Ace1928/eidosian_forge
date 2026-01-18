import functools
import itertools
from string import ascii_letters as ABC
from typing import List
import numpy as np
from scipy.sparse import csr_matrix
import pennylane as qml
from pennylane import BasisState, DeviceError, QubitDevice, StatePrep, Snapshot
from pennylane.devices.qubit import measure
from pennylane.operation import Operation
from pennylane.ops import Sum
from pennylane.ops.qubit.attributes import diagonal_in_z_basis
from pennylane.pulse import ParametrizedEvolution
from pennylane.measurements import ExpectationMP
from pennylane.typing import TensorLike
from pennylane.wires import WireError
from .._version import __version__
def classical_shadow(self, obs, circuit):
    """
        Returns the measured bits and recipes in the classical shadow protocol.

        The protocol is described in detail in the `classical shadows paper <https://arxiv.org/abs/2002.08953>`_.
        This measurement process returns the randomized Pauli measurements (the ``recipes``)
        that are performed for each qubit and snapshot as an integer:

        - 0 for Pauli X,
        - 1 for Pauli Y, and
        - 2 for Pauli Z.

        It also returns the measurement results (the ``bits``); 0 if the 1 eigenvalue
        is sampled, and 1 if the -1 eigenvalue is sampled.

        The device shots are used to specify the number of snapshots. If ``T`` is the number
        of shots and ``n`` is the number of qubits, then both the measured bits and the
        Pauli measurements have shape ``(T, n)``.

        This implementation leverages vectorization and offers a significant speed-up over
        the generic implementation.

        .. Note::

            This method internally calls ``np.einsum`` which supports at most 52 indices,
            thus the classical shadow measurement for this device supports at most 52
            qubits.

        .. seealso:: :func:`~pennylane.classical_shadow`

        Args:
            obs (~.pennylane.measurements.ClassicalShadowMP): The classical shadow measurement process
            circuit (~.tape.QuantumTape): The quantum tape that is being executed

        Returns:
            tensor_like[int]: A tensor with shape ``(2, T, n)``, where the first row represents
            the measured bits and the second represents the recipes used.
        """
    wires = obs.wires
    seed = obs.seed
    n_qubits = len(wires)
    n_snapshots = self.shots
    device_qubits = len(self.wires)
    mapped_wires = np.array(self.map_wires(wires))
    rng = np.random.RandomState(seed)
    recipes = rng.randint(0, 3, size=(n_snapshots, n_qubits))
    obs_list = self._stack([qml.X.compute_matrix(), qml.Y.compute_matrix(), qml.Z.compute_matrix()])
    uni_list = self._stack([qml.Hadamard.compute_matrix(), qml.Hadamard.compute_matrix() @ qml.RZ.compute_matrix(-np.pi / 2), qml.Identity.compute_matrix()])
    obs = obs_list[recipes]
    uni = uni_list[recipes]
    unmeasured_wires = [i for i in range(len(self.wires)) if i not in mapped_wires]
    transposed_state = np.transpose(self._state, axes=mapped_wires.tolist() + unmeasured_wires)
    outcomes = np.zeros((n_snapshots, n_qubits))
    stacked_state = self._stack([transposed_state for _ in range(n_snapshots)])
    for i in range(n_qubits):
        first_qubit_state = self._einsum(f'{ABC[device_qubits - i + 1]}{ABC[:device_qubits - i]},{ABC[device_qubits - i + 1]}{ABC[device_qubits - i]}{ABC[1:device_qubits - i]}->{ABC[device_qubits - i + 1]}a{ABC[device_qubits - i]}', stacked_state, self._conj(stacked_state))
        probs = (self._einsum('abc,acb->a', first_qubit_state, obs[:, i]) + 1) / 2
        samples = np.random.uniform(0, 1, size=probs.shape) > probs
        outcomes[:, i] = samples
        rotated_state = self._einsum('ab...,acb->ac...', stacked_state, uni[:, i])
        stacked_state = rotated_state[np.arange(n_snapshots), self._cast(samples, np.int8)]
        norms = np.sqrt(np.sum(np.abs(stacked_state) ** 2, tuple(range(1, device_qubits - i)), keepdims=True))
        stacked_state /= norms
    return self._cast(self._stack([outcomes, recipes]), dtype=np.int8)
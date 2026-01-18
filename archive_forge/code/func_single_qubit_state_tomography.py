import dataclasses
import itertools
from typing import Any, cast, Iterator, List, Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from cirq import circuits, ops, protocols
def single_qubit_state_tomography(sampler: 'cirq.Sampler', qubit: 'cirq.Qid', circuit: 'cirq.AbstractCircuit', repetitions: int=1000) -> TomographyResult:
    """Single-qubit state tomography.

    The density matrix of the output state of a circuit is measured by first
    doing projective measurements in the z-basis, which determine the
    diagonal elements of the matrix. A X/2 or Y/2 rotation is then added before
    the z-basis measurement, which determines the imaginary and real parts of
    the off-diagonal matrix elements, respectively.

    See Vandersypen and Chuang, Rev. Mod. Phys. 76, 1037 for details.

    Args:
        sampler: The quantum engine or simulator to run the circuits.
        qubit: The qubit under test.
        circuit: The circuit to execute on the qubit before tomography.
        repetitions: The number of measurements for each basis rotation.

    Returns:
        A TomographyResult object that stores and plots the density matrix.
    """
    circuit_z = circuit + circuits.Circuit(ops.measure(qubit, key='z'))
    results = sampler.run(circuit_z, repetitions=repetitions)
    rho_11 = np.mean(results.measurements['z'])
    rho_00 = 1.0 - rho_11
    circuit_x = circuits.Circuit(circuit, ops.X(qubit) ** 0.5, ops.measure(qubit, key='z'))
    results = sampler.run(circuit_x, repetitions=repetitions)
    rho_01_im = np.mean(results.measurements['z']) - 0.5
    circuit_y = circuits.Circuit(circuit, ops.Y(qubit) ** (-0.5), ops.measure(qubit, key='z'))
    results = sampler.run(circuit_y, repetitions=repetitions)
    rho_01_re = 0.5 - np.mean(results.measurements['z'])
    rho_01 = rho_01_re + 1j * rho_01_im
    rho_10 = np.conj(rho_01)
    rho = np.array([[rho_00, rho_01], [rho_10, rho_11]])
    return TomographyResult(rho)
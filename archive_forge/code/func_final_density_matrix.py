from typing import List, Optional, Sequence, Type, TYPE_CHECKING, Union
import numpy as np
from cirq import circuits, protocols, study, devices, ops, value
from cirq._doc import document
from cirq.sim import sparse_simulator, density_matrix_simulator
from cirq.sim.clifford import clifford_simulator
from cirq.transformers import measurement_transformers
def final_density_matrix(program: 'cirq.CIRCUIT_LIKE', *, noise: 'cirq.NOISE_MODEL_LIKE'=None, initial_state: 'cirq.STATE_VECTOR_LIKE'=0, param_resolver: 'cirq.ParamResolverOrSimilarType'=None, qubit_order: 'cirq.QubitOrderOrList'=ops.QubitOrder.DEFAULT, dtype: Type[np.complexfloating]=np.complex64, seed: Optional[Union[int, np.random.RandomState]]=None, ignore_measurement_results: bool=True) -> 'np.ndarray':
    """Returns the density matrix resulting from simulating the circuit.

    Note that, unlike `cirq.final_state_vector`, terminal measurements
    are not omitted. Instead, all measurements are treated as sources
    of decoherence (i.e. measurements do not collapse, they dephase). See
    ignore_measurement_results for details.

    Args:
        program: The circuit, gate, operation, or tree of operations
            to apply to the initial state in order to produce the result.
        noise: Noise model to use while running the simulation.
        param_resolver: Parameters to run with the program.
        qubit_order: Determines the canonical ordering of the qubits. This
            is often used in specifying the initial state, i.e. the
            ordering of the computational basis states.
        initial_state: If an int, the state is set to the computational
            basis state corresponding to this state. Otherwise  if this
            is a np.ndarray it is the full initial state. In this case it
            must be the correct size, be normalized (an L2 norm of 1), and
            be safely castable to an appropriate dtype for the simulator.
        dtype: The `numpy.dtype` used by the simulation. Typically one of
            `numpy.complex64` or `numpy.complex128`.
        seed: The random seed to use for this simulator.
        ignore_measurement_results: Defaults to True. When True, the returned
            density matrix is not conditioned on any measurement results.
            For example, this effectively replaces computational basis
            measurement with dephasing noise. The result density matrix in this
            case should be unique. When False, the result will be conditioned on
            sampled (but unreported) measurement results. In this case the
            result may vary from call to call.

    Returns:
        The density matrix for the state which results from applying the given
        operations to the desired initial state.

    """
    noise_model = devices.NoiseModel.from_noise_model_like(noise)
    circuit_like = _to_circuit(program)
    can_do_unitary_simulation = True
    if not protocols.has_unitary(circuit_like):
        can_do_unitary_simulation = False
    if isinstance(circuit_like, circuits.Circuit):
        if circuit_like.has_measurements():
            can_do_unitary_simulation = False
    if noise_model != devices.NO_NOISE:
        can_do_unitary_simulation = False
    if can_do_unitary_simulation:
        sparse_result = sparse_simulator.Simulator(dtype=dtype, seed=seed).simulate(program=circuit_like, initial_state=initial_state, qubit_order=qubit_order, param_resolver=param_resolver)
        return sparse_result.density_matrix_of()
    else:
        density_result = density_matrix_simulator.DensityMatrixSimulator(dtype=dtype, noise=noise, seed=seed).simulate(program=measurement_transformers.dephase_measurements(circuit_like) if ignore_measurement_results else circuit_like, initial_state=initial_state, qubit_order=qubit_order, param_resolver=param_resolver)
        return density_result.final_density_matrix
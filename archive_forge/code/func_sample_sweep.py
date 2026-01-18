from typing import List, Optional, Sequence, Type, TYPE_CHECKING, Union
import numpy as np
from cirq import circuits, protocols, study, devices, ops, value
from cirq._doc import document
from cirq.sim import sparse_simulator, density_matrix_simulator
from cirq.sim.clifford import clifford_simulator
from cirq.transformers import measurement_transformers
def sample_sweep(program: 'cirq.Circuit', params: 'cirq.Sweepable', *, noise: 'cirq.NOISE_MODEL_LIKE'=None, repetitions: int=1, dtype: Type[np.complexfloating]=np.complex64, seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE'=None) -> Sequence['cirq.Result']:
    """Runs the supplied Circuit, mimicking quantum hardware.

    In contrast to run, this allows for sweeping over different parameter
    values.

    Args:
        program: The circuit to simulate.
        params: Parameters to run with the program.
        noise: Noise model to use while running the simulation.
        repetitions: The number of repetitions to simulate, per set of
            parameter values.
        dtype: The `numpy.dtype` used by the simulation. Typically one of
            `numpy.complex64` or `numpy.complex128`.
            Favors speed over precision by default, i.e. uses `numpy.complex64`.
        seed: The random seed to use for this simulator.

    Returns:
        Result list for this run; one for each possible parameter
        resolver.
    """
    prng = value.parse_random_state(seed)
    trial_results: List[study.Result] = []
    for param_resolver in study.to_resolvers(params):
        measurements = sample(program, noise=noise, param_resolver=param_resolver, repetitions=repetitions, dtype=dtype, seed=prng)
        trial_results.append(measurements)
    return trial_results
from typing import Callable, Mapping, Optional, Sequence
import numpy as np
from cirq.circuits import Circuit
from cirq.ops import QubitOrder, QubitOrderOrList
from cirq.sim import final_state_vector
from cirq.value import state_vector_to_probabilities
def linear_xeb_fidelity_from_probabilities(hilbert_space_dimension: int, probabilities: Sequence[float]) -> float:
    """Linear XEB fidelity estimator.

    Estimates fidelity from ideal probabilities of observed bitstrings.

    This estimator makes two assumptions. First, it assumes that the circuit
    used in experiment is sufficiently scrambling that its output probabilities
    follow the Porter-Thomas distribution. This assumption holds for typical
    instances of random quantum circuits of sufficient depth. Second, it assumes
    that the circuit uses enough qubits so that the Porter-Thomas distribution
    can be approximated with the exponential distribution.

    In practice the validity of these assumptions can be confirmed by plotting
    a histogram of output probabilities and comparing it to the exponential
    distribution.

    The mean of this estimator is the true fidelity f and the variance is

        (1 + 2f - f^2) / M

    where f is the fidelity and M the number of observations, equal to
    len(probabilities). This is better than logarithmic XEB (see below)
    when fidelity is f < 0.32. Since this estimator is unbiased, the
    variance is equal to the mean squared error of the estimator.

    The estimator is intended for use with xeb_fidelity() below.

    Args:
        hilbert_space_dimension: Dimension of the Hilbert space on which
           the channel whose fidelity is being estimated is defined.
        probabilities: Ideal probabilities of bitstrings observed in
            experiment.
    Returns:
        Estimate of fidelity associated with an experimental realization
        of a quantum circuit.
    """
    return hilbert_space_dimension * np.mean(probabilities).item() - 1
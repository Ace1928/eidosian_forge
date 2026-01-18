from typing import Callable, Mapping, Optional, Sequence
import numpy as np
from cirq.circuits import Circuit
from cirq.ops import QubitOrder, QubitOrderOrList
from cirq.sim import final_state_vector
from cirq.value import state_vector_to_probabilities
def log_xeb_fidelity_from_probabilities(hilbert_space_dimension: int, probabilities: Sequence[float]) -> float:
    """Logarithmic XEB fidelity estimator.

    Estimates fidelity from ideal probabilities of observed bitstrings.

    See `linear_xeb_fidelity_from_probabilities` for the assumptions made
    by this estimator.

    The mean of this estimator is the true fidelity f and the variance is

        (pi^2/6 - f^2) / M

    where f is the fidelity and M the number of observations, equal to
    len(probabilities). This is better than linear XEB (see above) when
    fidelity is f > 0.32. Since this estimator is unbiased, the variance
    is equal to the mean squared error of the estimator.

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
    return np.log(hilbert_space_dimension) + np.euler_gamma + np.mean(np.log(probabilities))
from collections import Counter
from typing import Optional, Sequence
import warnings
from numpy.random import default_rng
import numpy as np
import pennylane as qml
from pennylane.measurements import (
from pennylane.typing import Result
from .initialize_state import create_initial_state
from .apply_operation import apply_operation
from .measure import measure
from .sampling import measure_with_samples
def gather_non_mcm(circuit_measurement, measurement, samples):
    """Combines, gathers and normalizes several measurements with trivial measurement values.

    Args:
        circuit_measurement (MeasurementProcess): measurement
        measurement (TensorLike): measurement results
        samples (List[dict]): Mid-circuit measurement samples

    Returns:
        TensorLike: The combined measurement outcome
    """
    if isinstance(circuit_measurement, CountsMP):
        return dict(sorted(measurement.items()))
    if isinstance(circuit_measurement, (ExpectationMP, ProbabilityMP)):
        return measurement / len(samples)
    if isinstance(circuit_measurement, SampleMP):
        return np.squeeze(np.concatenate(tuple((s.reshape(1, -1) for s in measurement))))
    return qml.math.var(np.concatenate(tuple((s.ravel() for s in measurement))))
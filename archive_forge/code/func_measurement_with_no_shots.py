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
def measurement_with_no_shots(measurement):
    return np.nan * np.ones_like(measurement.eigvals()) if isinstance(measurement, ProbabilityMP) else np.nan
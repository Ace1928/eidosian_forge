from pathlib import Path
from typing import List, Sequence
from warnings import warn
import numpy as np
from pennylane_lightning.core.lightning_base import (
def processing_fn_expval(tape):
    nonlocal ham
    num_params = len(tape.trainable_params)
    if num_params == 0:
        return np.array([], dtype=self.state.dtype)
    new_tape = tape.copy()
    new_tape._measurements = [qml.expval(ham)]
    return self.adjoint_jacobian(new_tape, starting_state, use_device_state)